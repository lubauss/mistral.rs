//! Quantized Qwen3-VL Vision Encoder for GGUF mmproj files.
//!
//! This module loads vision encoder weights from mmproj GGUF files
//! with tensor namespace `v.*` (vision) and `mm.*` (merger).
//!
//! Verified tensor structure from mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf:
//! - `v.blk.{0-26}.*` - Vision transformer blocks
//! - `v.deepstack.{8,16,24}.*` - Feature fusion layers
//! - `v.patch_embd.*` - Patch embedding
//! - `v.position_embd.weight` - Position embedding
//! - `v.post_ln.*` - Post layer norm
//! - `mm.{0,2}.*` - Multimodal projector (merger)

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Module;
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

use crate::attention::SdpaParams;
use crate::gguf::Content;
use crate::layers::Sdpa;

/// Simple layer normalization for vision encoder (with bias support).
/// OPTIMIZED: Pre-converts weights to F32 to avoid per-forward conversions.
#[derive(Debug, Clone)]
struct QLayerNorm {
    weight_f32: Tensor,      // Pre-converted to F32
    bias_f32: Option<Tensor>, // Pre-converted to F32
    eps: f64,
}

impl QLayerNorm {
    fn new(weight: Tensor, bias: Option<Tensor>, eps: f64) -> Result<Self> {
        // Pre-convert weights to F32 during construction (avoids per-forward overhead)
        let weight_f32 = if weight.dtype() == DType::F32 {
            weight
        } else {
            weight.to_dtype(DType::F32)?
        };
        let bias_f32 = bias
            .map(|b| {
                if b.dtype() == DType::F32 {
                    Ok(b)
                } else {
                    b.to_dtype(DType::F32)
                }
            })
            .transpose()?;
        Ok(Self { weight_f32, bias_f32, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_dtype = xs.dtype();
        // Only convert input if not already F32
        let xs = if x_dtype == DType::F32 {
            xs.clone()
        } else {
            xs.to_dtype(DType::F32)?
        };
        let hidden_size = xs.dim(D::Minus1)?;
        let mean = (xs.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let xs = xs.broadcast_sub(&mean)?;
        let var = ((&xs * &xs)?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let xs = xs.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        // Weights are already F32, no conversion needed
        let xs = xs.broadcast_mul(&self.weight_f32)?;
        let xs = if let Some(bias) = &self.bias_f32 {
            xs.broadcast_add(bias)?
        } else {
            xs
        };
        // Convert back to original dtype
        if x_dtype == DType::F32 {
            Ok(xs)
        } else {
            xs.to_dtype(x_dtype)
        }
    }
}

/// Configuration for the Qwen3-VL vision encoder.
/// Extracted from mmproj GGUF metadata (clip.vision.* namespace).
#[derive(Debug, Clone)]
pub struct Qwen3VLVisionConfig {
    pub block_count: usize,           // clip.vision.block_count (27)
    pub embedding_length: usize,      // clip.vision.embedding_length (1152)
    pub feed_forward_length: usize,   // clip.vision.feed_forward_length (4304)
    pub patch_size: usize,            // clip.vision.patch_size (16)
    pub image_size: usize,            // clip.vision.image_size (768)
    pub num_heads: usize,             // clip.vision.attention.head_count (16)
    pub spatial_merge_size: usize,    // clip.vision.spatial_merge_size (2)
    pub layer_norm_eps: f64,          // clip.vision.attention.layer_norm_epsilon
    pub projection_dim: usize,        // clip.vision.projection_dim (2048)
    pub deepstack_indices: Vec<usize>, // Layers 8, 16, 24 for deepstack
}

impl Default for Qwen3VLVisionConfig {
    fn default() -> Self {
        Self {
            block_count: 27,
            embedding_length: 1152,
            feed_forward_length: 4304,
            patch_size: 16,
            image_size: 768,
            num_heads: 16,
            spatial_merge_size: 2,
            layer_norm_eps: 1e-6,
            projection_dim: 2048,
            deepstack_indices: vec![8, 16, 24],
        }
    }
}

/// Quantized vision attention block.
/// Loads from: v.blk.{i}.attn_qkv.*, v.blk.{i}.attn_out.*
struct QVisionAttention {
    qkv: Arc<dyn QuantMethod>,     // v.blk.{i}.attn_qkv.weight/bias
    qkv_bias: Option<Tensor>,
    proj: Arc<dyn QuantMethod>,    // v.blk.{i}.attn_out.weight/bias
    proj_bias: Option<Tensor>,
    num_heads: usize,
    head_dim: usize,
}

impl QVisionAttention {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        cfg: &Qwen3VLVisionConfig,
        device: &Device,
    ) -> Result<Self> {
        let qkv_weight = ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
        let qkv_bias = ct.tensor(&format!("{prefix}.attn_qkv.bias"), device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let proj_weight = ct.tensor(&format!("{prefix}.attn_out.weight"), device)?;
        let proj_bias = ct.tensor(&format!("{prefix}.attn_out.bias"), device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let head_dim = cfg.embedding_length / cfg.num_heads;

        Ok(Self {
            qkv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(qkv_weight),
                b: qkv_bias.clone(),
            })?),
            qkv_bias,
            proj: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(proj_weight),
                b: proj_bias.clone(),
            })?),
            proj_bias,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let input_dtype = xs.dtype();
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;

        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let q = qkv.i(0)?.squeeze(0)?;
        let k = qkv.i(1)?.squeeze(0)?;
        let v = qkv.i(2)?.squeeze(0)?;

        // Apply rotary position embedding (RoPE needs F32 for numerical stability)
        // Only convert Q/K for RoPE, keep V in native dtype for SDPA
        let (q, k) = {
            let cos_f32 = if cos.dtype() == DType::F32 { cos.clone() } else { cos.to_dtype(DType::F32)? };
            let sin_f32 = if sin.dtype() == DType::F32 { sin.clone() } else { sin.to_dtype(DType::F32)? };
            let q_f32 = if q.dtype() == DType::F32 { q.clone() } else { q.to_dtype(DType::F32)? };
            let k_f32 = if k.dtype() == DType::F32 { k.clone() } else { k.to_dtype(DType::F32)? };
            let (q_rot, k_rot) = apply_rotary_pos_emb_vision(&q_f32, &k_f32, &cos_f32, &sin_f32)?;
            // Convert back to input dtype for SDPA (which supports BF16/F16 natively)
            (q_rot.to_dtype(input_dtype)?, k_rot.to_dtype(input_dtype)?)
        };

        // Process each sequence in the batch
        // Note: Q, K now in input_dtype after RoPE; V stays in original dtype from QKV projection
        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            // SDPA requires contiguous tensors after transpose
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            // Run attention (SDPA supports BF16/F16 natively on Metal)
            let chunk_out = Sdpa
                .run_attention(
                    &q_chunk.unsqueeze(0)?,
                    &k_chunk.unsqueeze(0)?,
                    &v_chunk.unsqueeze(0)?,
                    None,
                    None,
                    &SdpaParams {
                        n_kv_groups: 1,
                        sliding_window: None,
                        softcap: None,
                        softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
                    },
                )?
                .squeeze(0)?
                .transpose(0, 1)?
                .reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out);
        }
        let attn_output = Tensor::cat(&outputs, 0)?;
        self.proj.forward(&attn_output)
    }
}

/// Quantized vision MLP.
/// Loads from: v.blk.{i}.ffn_up.*, v.blk.{i}.ffn_down.*
struct QVisionMlp {
    fc_up: Arc<dyn QuantMethod>,    // v.blk.{i}.ffn_up.weight/bias
    fc_down: Arc<dyn QuantMethod>,  // v.blk.{i}.ffn_down.weight/bias
}

impl QVisionMlp {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let up_weight = ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
        let up_bias = ct.tensor(&format!("{prefix}.ffn_up.bias"), device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let down_weight = ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
        let down_bias = ct.tensor(&format!("{prefix}.ffn_down.bias"), device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        Ok(Self {
            fc_up: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(up_weight),
                b: up_bias,
            })?),
            fc_down: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(down_weight),
                b: down_bias,
            })?),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc_up.forward(xs)?;
        let xs = xs.gelu_erf()?; // GELU activation
        self.fc_down.forward(&xs)
    }
}

/// Quantized vision transformer block.
/// Loads from: v.blk.{i}.*
struct QVisionBlock {
    attn: QVisionAttention,
    mlp: QVisionMlp,
    ln1: QLayerNorm,  // v.blk.{i}.ln1.*
    ln2: QLayerNorm,  // v.blk.{i}.ln2.*
}

impl QVisionBlock {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        layer_idx: usize,
        cfg: &Qwen3VLVisionConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("v.blk.{layer_idx}");

        let attn = QVisionAttention::load(ct, &prefix, cfg, device)?;
        let mlp = QVisionMlp::load(ct, &prefix, device)?;

        let ln1 = QLayerNorm::new(
            ct.tensor(&format!("{prefix}.ln1.weight"), device)?.dequantize(device)?,
            Some(ct.tensor(&format!("{prefix}.ln1.bias"), device)?.dequantize(device)?),
            cfg.layer_norm_eps,
        )?;

        let ln2 = QLayerNorm::new(
            ct.tensor(&format!("{prefix}.ln2.weight"), device)?.dequantize(device)?,
            Some(ct.tensor(&format!("{prefix}.ln2.bias"), device)?.dequantize(device)?),
            cfg.layer_norm_eps,
        )?;

        Ok(Self { attn, mlp, ln1, ln2 })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.ln1.forward(xs)?;
        let xs = self.attn.forward(&xs, cu_seqlens, cos, sin)?;
        let xs = (residual + xs)?;

        let residual = xs.clone();
        let xs = self.ln2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// Quantized deepstack layer for feature fusion.
/// Loads from: v.deepstack.{8,16,24}.*
/// Uses spatial merge to combine patches before projection.
struct QDeepstackLayer {
    fc1: Arc<dyn QuantMethod>,  // v.deepstack.{i}.fc1.*
    fc2: Arc<dyn QuantMethod>,  // v.deepstack.{i}.fc2.*
    norm: QLayerNorm,           // v.deepstack.{i}.norm.*
    spatial_merge_unit: usize,  // spatial_merge_size^2 = 4
    hidden_size: usize,         // 1024
    merged_hidden_size: usize,  // hidden_size * spatial_merge_unit = 4096
}

impl QDeepstackLayer {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        layer_idx: usize,
        cfg: &Qwen3VLVisionConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("v.deepstack.{layer_idx}");

        let fc1_weight = ct.tensor(&format!("{prefix}.fc1.weight"), device)?;
        let fc1_bias = ct.tensor(&format!("{prefix}.fc1.bias"), device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let fc2_weight = ct.tensor(&format!("{prefix}.fc2.weight"), device)?;
        let fc2_bias = ct.tensor(&format!("{prefix}.fc2.bias"), device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let norm = QLayerNorm::new(
            ct.tensor(&format!("{prefix}.norm.weight"), device)?.dequantize(device)?,
            Some(ct.tensor(&format!("{prefix}.norm.bias"), device)?.dequantize(device)?),
            cfg.layer_norm_eps,
        )?;

        let spatial_merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merged_hidden_size = cfg.embedding_length * spatial_merge_unit;

        Ok(Self {
            fc1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(fc1_weight),
                b: fc1_bias,
            })?),
            fc2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(fc2_weight),
                b: fc2_bias,
            })?),
            norm,
            spatial_merge_unit,
            hidden_size: cfg.embedding_length,
            merged_hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            candle_core::bail!(
                "Sequence length {} is not divisible by spatial merge unit {}",
                seq_len,
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;

        // Reshape to merge spatial patches: [seq_len, hidden] -> [grouped, merged_hidden]
        let xs = xs.reshape((grouped, self.merged_hidden_size))?;
        let xs = self.norm.forward(&xs)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = xs.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

/// Quantized multimodal projector (merger).
/// Loads from: mm.0.*, mm.2.*
/// Note: Norm is applied separately via post_ln before calling merger.forward()
struct QMerger {
    fc1: Arc<dyn QuantMethod>,  // mm.0.*
    fc2: Arc<dyn QuantMethod>,  // mm.2.*
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
}

impl QMerger {
    fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        cfg: &Qwen3VLVisionConfig,
        device: &Device,
    ) -> Result<Self> {
        let fc1_weight = ct.tensor("mm.0.weight", device)?;
        let fc1_bias = ct.tensor("mm.0.bias", device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let fc2_weight = ct.tensor("mm.2.weight", device)?;
        let fc2_bias = ct.tensor("mm.2.bias", device)
            .ok()
            .map(|t| t.dequantize(device))
            .transpose()?;

        let spatial_merge_unit = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merged_hidden_size = cfg.embedding_length * spatial_merge_unit;

        Ok(Self {
            fc1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(fc1_weight),
                b: fc1_bias,
            })?),
            fc2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(fc2_weight),
                b: fc2_bias,
            })?),
            spatial_merge_unit,
            merged_hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            candle_core::bail!(
                "Merger: Sequence length {} is not divisible by spatial merge unit {}",
                seq_len,
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;

        // Reshape to merge spatial patches: [seq_len, hidden] -> [grouped, merged_hidden]
        // Note: Norm was already applied via post_ln before this call
        let xs = xs.reshape((grouped, self.merged_hidden_size))?;

        let xs = self.fc1.forward(&xs)?;
        let xs = xs.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

/// Complete quantized Qwen3-VL vision encoder.
/// Loaded from mmproj GGUF file.
pub struct QQwen3VLVisionEncoder {
    patch_embed_weight: Tensor,    // v.patch_embd.weight (conv kernel)
    patch_embed_bias: Tensor,      // v.patch_embd.bias
    position_embed: Tensor,        // v.position_embd.weight
    blocks: Vec<QVisionBlock>,     // v.blk.{0-26}.*
    deepstack: HashMap<usize, QDeepstackLayer>,  // v.deepstack.{8,16,24}.*
    post_ln: QLayerNorm,           // v.post_ln.*
    merger: QMerger,               // mm.*
    config: Qwen3VLVisionConfig,
    device: Device,
}

impl QQwen3VLVisionEncoder {
    /// Load the vision encoder from an mmproj GGUF Content.
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        device: &Device,
    ) -> Result<Self> {
        // Extract config from GGUF metadata
        let meta = ct.get_metadata();
        let mut config = Qwen3VLVisionConfig {
            block_count: meta.get("clip.vision.block_count")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(27),
            embedding_length: meta.get("clip.vision.embedding_length")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(1152),
            feed_forward_length: meta.get("clip.vision.feed_forward_length")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(4304),
            patch_size: meta.get("clip.vision.patch_size")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(16),
            image_size: meta.get("clip.vision.image_size")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(768),
            num_heads: meta.get("clip.vision.attention.head_count")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(16),
            spatial_merge_size: meta.get("clip.vision.spatial_merge_size")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(2),
            layer_norm_eps: meta.get("clip.vision.attention.layer_norm_epsilon")
                .and_then(|v| v.to_f32().ok())
                .map(|v| v as f64)
                .unwrap_or(1e-6),
            projection_dim: meta.get("clip.vision.projection_dim")
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(2048),
            deepstack_indices: vec![],  // Will be discovered dynamically below
        };

        // Dynamically discover deepstack indices by scanning for existing tensors
        let block_count = config.block_count;
        let mut deepstack_indices = Vec::new();
        for i in 0..block_count {
            if ct.has_tensor(&format!("v.deepstack.{i}.fc1.weight")) {
                deepstack_indices.push(i);
            }
        }
        if !deepstack_indices.is_empty() {
            tracing::info!(
                "Discovered deepstack layers at indices: {:?}",
                deepstack_indices
            );
        }
        config.deepstack_indices = deepstack_indices;

        tracing::info!(
            "Loading Qwen3-VL vision encoder: {} blocks, {}d embed, {} heads",
            config.block_count,
            config.embedding_length,
            config.num_heads
        );

        // Load patch embedding (conv weights)
        let patch_embed_weight = ct.tensor("v.patch_embd.weight", device)?.dequantize(device)?;
        tracing::info!("Patch embed weight shape: {:?}", patch_embed_weight.dims());
        let patch_embed_bias = ct.tensor("v.patch_embd.bias", device)?.dequantize(device)?;
        tracing::info!("Patch embed bias shape: {:?}", patch_embed_bias.dims());

        // Load position embedding
        let position_embed = ct.tensor("v.position_embd.weight", device)?.dequantize(device)?;

        // Load vision transformer blocks
        let mut blocks = Vec::with_capacity(config.block_count);
        for i in 0..config.block_count {
            blocks.push(QVisionBlock::load(ct, i, &config, device)?);
        }

        // Load deepstack layers
        let mut deepstack = HashMap::new();
        for &idx in &config.deepstack_indices {
            deepstack.insert(idx, QDeepstackLayer::load(ct, idx, &config, device)?);
        }

        // Load post layer norm
        let post_ln_weight = ct.tensor("v.post_ln.weight", device)?.dequantize(device)?;
        let post_ln_bias = ct.tensor("v.post_ln.bias", device)?.dequantize(device)?;
        tracing::info!("Post LN weight shape: {:?}", post_ln_weight.dims());
        tracing::info!("Post LN bias shape: {:?}", post_ln_bias.dims());
        let post_ln = QLayerNorm::new(
            post_ln_weight,
            Some(post_ln_bias),
            config.layer_norm_eps,
        )?;

        // Load merger
        let merger = QMerger::load(ct, &config, device)?;

        Ok(Self {
            patch_embed_weight,
            patch_embed_bias,
            position_embed,
            blocks,
            deepstack,
            post_ln,
            merger,
            config,
            device: device.clone(),
        })
    }

    /// Get the output dimension of the vision encoder.
    pub fn output_dim(&self) -> usize {
        self.config.projection_dim
    }

    /// Get the config.
    pub fn config(&self) -> &Qwen3VLVisionConfig {
        &self.config
    }

    /// Get the spatial merge size
    pub fn spatial_merge_size(&self) -> usize {
        self.config.spatial_merge_size
    }

    /// Compute number of grid positions per side for position embeddings
    fn num_grid_per_side(&self) -> usize {
        // Position embedding is stored as [num_positions, hidden_size]
        // num_positions is typically a perfect square (e.g., 3136 = 56*56)
        let num_positions = self.position_embed.dim(0).unwrap_or(3136);
        (num_positions as f64).sqrt().round() as usize
    }

    /// Build rotary position embeddings for vision transformer
    fn make_rot_pos_emb(&self, seqlen: usize) -> Result<Tensor> {
        let head_dim = self.config.embedding_length / self.config.num_heads;
        let theta: f32 = 10000.0;
        let half_dim = head_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / half_dim as f32))
            .collect();

        let inv_freq = Tensor::from_vec(inv_freq, (1, half_dim), &self.device)?;
        let seq = Tensor::arange(0f32, seqlen as f32, &self.device)?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&inv_freq)
    }

    /// Compute 2D rotary position embeddings for vision (GPU-optimized)
    ///
    /// Generates coordinate indices entirely on GPU to avoid CPU-GPU sync bottleneck.
    /// Pattern: for br in 0..merged_h, for bc in 0..merged_w, for ir in 0..merge_size, for ic in 0..merge_size
    /// row = br * merge_size + ir, col = bc * merge_size + ic
    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_hw = grid
            .iter()
            .flat_map(|v| v[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;

        let freq_table = self.make_rot_pos_emb(max_hw)?;
        let merge_size = self.config.spatial_merge_size;
        let device = &self.device;

        let mut all_row_indices: Vec<Tensor> = Vec::new();
        let mut all_col_indices: Vec<Tensor> = Vec::new();

        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / merge_size;
            let merged_w = w / merge_size;

            // GPU-side coordinate generation
            // Loop order: br (outer) -> bc -> ir -> ic (inner)
            let repeat_per_br = merged_w * merge_size * merge_size;
            let repeat_per_bc = merge_size * merge_size;

            // br: each value [0..merged_h) repeated repeat_per_br times
            // [0,0,0,0,..., 1,1,1,1,..., ..., merged_h-1,...]
            let br = Tensor::arange(0i64, merged_h as i64, device)?
                .unsqueeze(1)? // [merged_h, 1]
                .repeat((1, repeat_per_br))? // [merged_h, repeat_per_br]
                .flatten_all()?; // [merged_h * repeat_per_br]

            // bc: each value [0..merged_w) repeated repeat_per_bc times, tiled merged_h times
            // [0,0,0,0, 1,1,1,1, ..., merged_w-1,...] repeated merged_h times
            let bc_row = Tensor::arange(0i64, merged_w as i64, device)?
                .unsqueeze(1)? // [merged_w, 1]
                .repeat((1, repeat_per_bc))? // [merged_w, repeat_per_bc]
                .flatten_all()?; // [merged_w * repeat_per_bc]
            let bc = bc_row
                .unsqueeze(0)? // [1, merged_w * repeat_per_bc]
                .repeat((merged_h, 1))? // [merged_h, merged_w * repeat_per_bc]
                .flatten_all()?; // [total_spatial]

            // ir: each value [0..merge_size) repeated merge_size times (for ic), tiled for all blocks
            // [0,0,1,1, 0,0,1,1, ...] for merge_size=2
            let ir_block = Tensor::arange(0i64, merge_size as i64, device)?
                .unsqueeze(1)? // [merge_size, 1]
                .repeat((1, merge_size))? // [merge_size, merge_size]
                .flatten_all()?; // [merge_size^2]
            let ir = ir_block
                .unsqueeze(0)? // [1, merge_size^2]
                .repeat((merged_h * merged_w, 1))? // [merged_h * merged_w, merge_size^2]
                .flatten_all()?; // [total_spatial]

            // ic: [0..merge_size) tiled for everything
            // [0,1, 0,1, 0,1, ...] for merge_size=2
            let ic_single = Tensor::arange(0i64, merge_size as i64, device)?; // [merge_size]
            let ic = ic_single
                .unsqueeze(0)? // [1, merge_size]
                .repeat((merged_h * merged_w * merge_size, 1))? // [n, merge_size]
                .flatten_all()?; // [total_spatial]

            // Compute coordinates: row = br * merge_size + ir, col = bc * merge_size + ic
            // Use affine(scale, bias) for scalar multiplication: tensor * scale + bias
            let row_coords = br
                .to_dtype(DType::F32)?
                .affine(merge_size as f64, 0.)?
                .broadcast_add(&ir.to_dtype(DType::F32)?)?
                .to_dtype(DType::I64)?;
            let col_coords = bc
                .to_dtype(DType::F32)?
                .affine(merge_size as f64, 0.)?
                .broadcast_add(&ic.to_dtype(DType::F32)?)?
                .to_dtype(DType::I64)?;

            // Repeat for temporal dimension
            let row_coords = if t > 1 {
                row_coords
                    .unsqueeze(0)? // [1, spatial]
                    .repeat((t, 1))? // [t, spatial]
                    .flatten_all()? // [t * spatial]
            } else {
                row_coords
            };
            let col_coords = if t > 1 {
                col_coords
                    .unsqueeze(0)?
                    .repeat((t, 1))?
                    .flatten_all()?
            } else {
                col_coords
            };

            all_row_indices.push(row_coords);
            all_col_indices.push(col_coords);
        }

        // Concatenate all grids
        let rows = Tensor::cat(&all_row_indices, 0)?;
        let cols = Tensor::cat(&all_col_indices, 0)?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;

        let total_tokens = rows.dim(0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
    }

    /// Build cumulative sequence lengths for attention
    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|v| v[0] as usize).sum::<usize>() + 1);
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    /// Interpolate position embeddings for variable resolution
    fn interpolate_pos_embed(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let dtype = self.position_embed.dtype();
        let grid = grid_thw.to_vec2::<u32>()?;
        let num_grid = self.num_grid_per_side();
        let hidden_size = self.config.embedding_length;
        let merge_size = self.config.spatial_merge_size;

        // Simple bilinear interpolation for position embeddings
        let linspace = |steps: usize| -> Vec<f32> {
            if steps == 1 {
                return vec![0.0];
            }
            let max_val = (num_grid - 1) as f32;
            let step = max_val / (steps.saturating_sub(1)) as f32;
            (0..steps).map(|i| i as f32 * step).collect()
        };

        let mut all_embeds = Vec::new();
        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;

            let h_vals = linspace(h);
            let w_vals = linspace(w);

            let mut pos_indices = Vec::with_capacity(h * w);
            let mut weights = Vec::with_capacity(h * w * 4);

            for &hv in &h_vals {
                for &wv in &w_vals {
                    let h_floor = hv.floor() as usize;
                    let w_floor = wv.floor() as usize;
                    let h_ceil = (hv.ceil() as usize).min(num_grid - 1);
                    let w_ceil = (wv.ceil() as usize).min(num_grid - 1);

                    let dh = hv - h_floor as f32;
                    let dw = wv - w_floor as f32;

                    // Indices for 4 corners
                    let idx00 = h_floor * num_grid + w_floor;
                    let idx01 = h_floor * num_grid + w_ceil;
                    let idx10 = h_ceil * num_grid + w_floor;
                    let idx11 = h_ceil * num_grid + w_ceil;

                    pos_indices.push((idx00, idx01, idx10, idx11));
                    weights.push(((1.0 - dh) * (1.0 - dw), (1.0 - dh) * dw, dh * (1.0 - dw), dh * dw));
                }
            }

            // For each position, interpolate from the 4 corners
            let mut interpolated = Vec::with_capacity(h * w);
            for ((idx00, idx01, idx10, idx11), (w00, w01, w10, w11)) in pos_indices.iter().zip(weights.iter()) {
                let p00 = self.position_embed.i(*idx00)?;
                let p01 = self.position_embed.i(*idx01)?;
                let p10 = self.position_embed.i(*idx10)?;
                let p11 = self.position_embed.i(*idx11)?;

                let interp = ((p00 * *w00 as f64)? + (p01 * *w01 as f64)? + (p10 * *w10 as f64)? + (p11 * *w11 as f64)?)?;
                interpolated.push(interp);
            }

            // Stack and repeat for temporal dimension
            let pos_embed_hw = Tensor::stack(&interpolated.iter().collect::<Vec<_>>(), 0)?;
            let pos_embed_thw = pos_embed_hw.repeat((t, 1))?;

            // Permute for spatial merge
            let pos_embed = pos_embed_thw.reshape((
                t,
                h / merge_size,
                merge_size,
                w / merge_size,
                merge_size,
                hidden_size,
            ))?;
            let pos_embed = pos_embed
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((t * h * w, hidden_size))?;
            all_embeds.push(pos_embed);
        }

        Tensor::cat(&all_embeds, 0)?.to_dtype(dtype)
    }

    /// Apply 2D patch embedding (convolution)
    fn patch_embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs shape: [num_patches, patch_pixels] where patch_pixels = channels * temporal * H * W
        // weight shape: [out_channels, in_channels, patch_h, patch_w] (4D conv kernel)
        // GGUF mmproj uses 2D conv, so temporal is folded into batch dimension
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let weight = self.patch_embed_weight.to_dtype(DType::F32)?;
        let bias = self.patch_embed_bias.to_dtype(DType::F32)?;

        // Flatten conv weight from [out_channels, in_channels, H, W] to [out_channels, in_channels*H*W]
        let weight_shape = weight.dims();
        let out_channels = weight_shape[0];
        let in_features: usize = weight_shape[1..].iter().product();
        let weight_flat = weight.reshape((out_channels, in_features))?;

        // Input has shape [num_patches, in_features * temporal] where temporal features are interleaved
        // For GGUF 2D conv (converted from 3D), we process each temporal slice and sum
        let xs_shape = xs.dims();
        let num_patches = xs_shape[0];
        let total_features = xs_shape[1];

        // Determine temporal factor from mismatch between input features and weight features
        let temporal = total_features / in_features;
        if total_features % in_features != 0 {
            candle_core::bail!(
                "Input features {} not divisible by weight in_features {}",
                total_features,
                in_features
            );
        }

        // OPTIMIZED: Batch matmul instead of temporal loop
        // Reshape to [num_patches * temporal, in_features] for single matmul
        let xs_flat = xs.reshape((num_patches * temporal, in_features))?;

        // Single matmul: [num_patches * temporal, in_features] @ [in_features, out_channels]
        let weight_t = weight_flat.t()?;
        let projected = xs_flat.matmul(&weight_t)?; // [num_patches * temporal, out_channels]

        // Reshape and sum over temporal dimension
        // [num_patches * temporal, out_channels] -> [num_patches, temporal, out_channels] -> sum -> [num_patches, out_channels]
        let out = projected
            .reshape((num_patches, temporal, out_channels))?
            .sum(1)?; // Sum over temporal dimension

        // Apply bias and return
        out.broadcast_add(&bias)?.to_dtype(dtype)
    }

    /// Forward pass through the vision encoder.
    ///
    /// Args:
    ///     pixel_values: Tensor of shape [batch, num_images, num_patches, features] or [total_patches, features]
    ///     grid_thw: Tensor of shape [num_images, 3] with (temporal, height, width) for each image
    ///
    /// Returns:
    ///     (merged_embeddings, deepstack_features)
    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        let dtype = self.position_embed.dtype();

        // Flatten pixel_values from [batch, num_images, num_patches, features] to [total_patches, features]
        let pixel_values = match pixel_values.dims().len() {
            2 => pixel_values.clone(), // Already [total_patches, features]
            4 => {
                // [batch, num_images, num_patches, features] -> [total_patches, features]
                let (b, n, p, f) = pixel_values.dims4()?;
                pixel_values.reshape((b * n * p, f))?
            }
            3 => {
                // [num_images, num_patches, features] -> [total_patches, features]
                let (n, p, f) = pixel_values.dims3()?;
                pixel_values.reshape((n * p, f))?
            }
            _ => {
                candle_core::bail!(
                    "Unexpected pixel_values shape {:?}, expected 2, 3, or 4 dimensions",
                    pixel_values.dims()
                );
            }
        };

        // 1. Patch embedding
        let xs = self.patch_embed_forward(&pixel_values.to_dtype(dtype)?)?;

        // 2. Position embedding
        let pos_embeds = self.interpolate_pos_embed(grid_thw)?;
        let mut hidden_states = xs.add(&pos_embeds)?;

        // 3. Rotary position embedding for attention
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let head_dim = self.config.embedding_length / self.config.num_heads;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, head_dim))?;
        // cos/sin shape should be [seq_len, head_dim] to match q/k [seq_len, num_heads, head_dim]
        let cos = rotary_pos_emb.cos()?.to_dtype(DType::F32)?;
        let sin = rotary_pos_emb.sin()?.to_dtype(DType::F32)?;

        // 4. Build cumulative sequence lengths
        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        // 5. Process through transformer blocks with deepstack
        let mut deepstack_features = Vec::new();
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;

            // Check if this layer has a deepstack merger
            if let Some(deepstack_layer) = self.deepstack.get(&layer_idx) {
                let feat = deepstack_layer.forward(&hidden_states)?;
                deepstack_features.push(feat);
            }
        }

        // 6. Post layer norm
        hidden_states = self.post_ln.forward(&hidden_states)?;

        // 7. Merger (projects to output dimension)
        let merged = self.merger.forward(&hidden_states)?;

        Ok((merged, deepstack_features))
    }
}

// Helper functions for rotary position embedding
fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = Qwen3VLVisionConfig::default();
        assert_eq!(cfg.block_count, 27);
        assert_eq!(cfg.embedding_length, 1152);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.deepstack_indices, vec![8, 16, 24]);
    }
}
