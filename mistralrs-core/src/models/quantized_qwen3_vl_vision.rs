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
        let debug_weights = std::env::var("MISTRALRS_DEBUG_WEIGHTS").is_ok();

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

        // Debug: Print projection weight statistics for block 0
        if debug_weights && prefix == "v.blk.0" {
            let proj_dequant = proj_weight.dequantize(device)?;
            let proj_f32 = proj_dequant.to_dtype(candle_core::DType::F32)?;
            let proj_mean = proj_f32.mean_all()?.to_scalar::<f32>()?;
            let proj_var = proj_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            let proj_min = proj_f32.flatten_all()?.min(0)?.to_scalar::<f32>()?;
            let proj_max = proj_f32.flatten_all()?.max(0)?.to_scalar::<f32>()?;
            eprintln!("[DEBUG GGUF WEIGHTS] {}.attn_out.weight:", prefix);
            eprintln!("  shape: {:?}, dtype: {:?}", proj_dequant.dims(), proj_dequant.dtype());
            eprintln!("  mean={:.6}, std={:.6}, min={:.6}, max={:.6}", proj_mean, proj_var.sqrt(), proj_min, proj_max);

            if let Some(ref bias) = proj_bias {
                let bias_f32 = bias.to_dtype(candle_core::DType::F32)?;
                let bias_mean = bias_f32.mean_all()?.to_scalar::<f32>()?;
                let bias_var = bias_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
                eprintln!("[DEBUG GGUF WEIGHTS] {}.attn_out.bias:", prefix);
                eprintln!("  shape: {:?}, mean={:.6}, std={:.6}", bias.dims(), bias_mean, bias_var.sqrt());
            } else {
                eprintln!("[DEBUG GGUF WEIGHTS] {}.attn_out.bias: NONE", prefix);
            }
        }

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
        let debug = std::env::var("MISTRALRS_DEBUG_ATTN").is_ok();
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;

        if debug {
            let h_f32 = hidden_states.to_dtype(candle_core::DType::F32)?;
            let h_mean = h_f32.mean_all()?.to_scalar::<f32>()?;
            let h_std = h_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG ATTN] QKV output: mean={:.6}, std={:.6}", h_mean, h_std);
        }

        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let q = qkv.i(0)?.squeeze(0)?;
        let k = qkv.i(1)?.squeeze(0)?;
        let v = qkv.i(2)?.squeeze(0)?;

        if debug {
            let q_f32 = q.to_dtype(candle_core::DType::F32)?;
            let k_f32 = k.to_dtype(candle_core::DType::F32)?;
            let v_f32 = v.to_dtype(candle_core::DType::F32)?;
            eprintln!("[DEBUG ATTN] Q: mean={:.6}, std={:.6}", q_f32.mean_all()?.to_scalar::<f32>()?, q_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
            eprintln!("[DEBUG ATTN] K: mean={:.6}, std={:.6}", k_f32.mean_all()?.to_scalar::<f32>()?, k_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
            eprintln!("[DEBUG ATTN] V: mean={:.6}, std={:.6}", v_f32.mean_all()?.to_scalar::<f32>()?, v_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
        }

        // Apply rotary position embedding
        // OPTIMIZED: RoPE in native dtype (F16/BF16) - Metal supports all ops
        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        if debug {
            let q_f32 = q.to_dtype(candle_core::DType::F32)?;
            let k_f32 = k.to_dtype(candle_core::DType::F32)?;
            eprintln!("[DEBUG ATTN] After RoPE - Q: mean={:.6}, std={:.6}", q_f32.mean_all()?.to_scalar::<f32>()?, q_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
            eprintln!("[DEBUG ATTN] After RoPE - K: mean={:.6}, std={:.6}", k_f32.mean_all()?.to_scalar::<f32>()?, k_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
        }

        let sdpa_params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
        };

        // OPTIMIZED: Fast path for single sequence (most common case for single image)
        // Avoids loop overhead, Vec allocation, and Tensor::cat
        let attn_output = if cu_seqlens.len() == 2 {
            // Single sequence: process directly without loop
            // Shape: [seq_len, num_heads, head_dim] -> [1, num_heads, seq_len, head_dim]
            let q_t = q.transpose(0, 1)?.unsqueeze(0)?;
            let k_t = k.transpose(0, 1)?.unsqueeze(0)?;
            let v_t = v.transpose(0, 1)?.unsqueeze(0)?;

            Sdpa
                .run_attention(&q_t, &k_t, &v_t, None, None, &sdpa_params)?
                .squeeze(0)?
                .transpose(0, 1)?
                .reshape((seq_len, self.num_heads * self.head_dim))?
        } else {
            // Multiple sequences: process each separately
            // This path is rare for single-image inference
            let mut outputs = Vec::with_capacity(cu_seqlens.len() - 1);
            for window in cu_seqlens.windows(2) {
                let start = window[0];
                let end = window[1];
                if end <= start {
                    continue;
                }
                let len = end - start;

                // OPTIMIZED: Transpose once, then narrow (avoids contiguous on full tensor)
                let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.unsqueeze(0)?;
                let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.unsqueeze(0)?;
                let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.unsqueeze(0)?;

                let chunk_out = Sdpa
                    .run_attention(&q_chunk, &k_chunk, &v_chunk, None, None, &sdpa_params)?
                    .squeeze(0)?
                    .transpose(0, 1)?
                    .reshape((len, self.num_heads * self.head_dim))?;
                outputs.push(chunk_out);
            }
            Tensor::cat(&outputs, 0)?
        };

        if debug {
            let a_f32 = attn_output.to_dtype(candle_core::DType::F32)?;
            eprintln!("[DEBUG ATTN] SDPA output: mean={:.6}, std={:.6}", a_f32.mean_all()?.to_scalar::<f32>()?, a_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
        }

        let output = self.proj.forward(&attn_output)?;

        if debug {
            let o_f32 = output.to_dtype(candle_core::DType::F32)?;
            eprintln!("[DEBUG ATTN] Proj output: mean={:.6}, std={:.6}", o_f32.mean_all()?.to_scalar::<f32>()?, o_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt());
        }

        Ok(output)
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
        // OPTIMIZED: Use references for residual instead of clone where possible
        // First residual block: attention
        let normed = self.ln1.forward(xs)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;
        let xs = (xs + attn_out)?;  // xs is consumed here, no clone needed

        // Second residual block: MLP
        let normed = self.ln2.forward(&xs)?;
        let mlp_out = self.mlp.forward(&normed)?;
        &xs + mlp_out  // Return reference add to avoid clone
    }

    #[allow(dead_code)]
    fn forward_debug(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let debug = std::env::var("MISTRALRS_DEBUG_VISION_BLOCKS").is_ok() && layer_idx == 0;

        if debug {
            let xs_f32 = xs.to_dtype(candle_core::DType::F32)?;
            let xs_mean = xs_f32.mean_all()?.to_scalar::<f32>()?;
            let xs_std = xs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG BLOCK {}] Input: mean={:.6}, std={:.6}", layer_idx, xs_mean, xs_std);
        }

        let normed = self.ln1.forward(xs)?;
        if debug {
            let n_f32 = normed.to_dtype(candle_core::DType::F32)?;
            let n_mean = n_f32.mean_all()?.to_scalar::<f32>()?;
            let n_std = n_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG BLOCK {}] After LN1: mean={:.6}, std={:.6}", layer_idx, n_mean, n_std);
        }

        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;
        if debug {
            let a_f32 = attn_out.to_dtype(candle_core::DType::F32)?;
            let a_mean = a_f32.mean_all()?.to_scalar::<f32>()?;
            let a_std = a_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG BLOCK {}] Attn output: mean={:.6}, std={:.6}", layer_idx, a_mean, a_std);
        }

        let xs = (xs + attn_out)?;
        if debug {
            let xs_f32 = xs.to_dtype(candle_core::DType::F32)?;
            let xs_mean = xs_f32.mean_all()?.to_scalar::<f32>()?;
            let xs_std = xs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG BLOCK {}] After attn residual: mean={:.6}, std={:.6}", layer_idx, xs_mean, xs_std);
        }

        let normed = self.ln2.forward(&xs)?;
        let mlp_out = self.mlp.forward(&normed)?;
        if debug {
            let m_f32 = mlp_out.to_dtype(candle_core::DType::F32)?;
            let m_mean = m_f32.mean_all()?.to_scalar::<f32>()?;
            let m_std = m_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG BLOCK {}] MLP output: mean={:.6}, std={:.6}", layer_idx, m_mean, m_std);
        }

        let result = (&xs + mlp_out)?;
        if debug {
            let r_f32 = result.to_dtype(candle_core::DType::F32)?;
            let r_mean = r_f32.mean_all()?.to_scalar::<f32>()?;
            let r_std = r_f32.var(0)?.mean_all()?.to_scalar::<f32>()?.sqrt();
            eprintln!("[DEBUG BLOCK {}] Output: mean={:.6}, std={:.6}", layer_idx, r_mean, r_std);
        }

        Ok(result)
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
    patch_embed_weight_0: Tensor,  // v.patch_embd.weight (temporal slice 0)
    patch_embed_weight_1: Tensor,  // v.patch_embd.weight.1 (temporal slice 1)
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
        // GGUF mmproj splits Conv3D into two Conv2D weights for temporal_patch_size=2
        let patch_embed_weight_0 = ct.tensor("v.patch_embd.weight", device)?.dequantize(device)?;
        tracing::info!("Patch embed weight_0 shape: {:?}", patch_embed_weight_0.dims());
        let patch_embed_weight_1 = ct.tensor("v.patch_embd.weight.1", device)?.dequantize(device)?;
        tracing::info!("Patch embed weight_1 shape: {:?}", patch_embed_weight_1.dims());
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
            patch_embed_weight_0,
            patch_embed_weight_1,
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
    /// OPTIMIZED: Uses batched tensor operations instead of per-position loops
    fn interpolate_pos_embed(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let dtype = self.position_embed.dtype();
        let device = self.position_embed.device();
        let grid = grid_thw.to_vec2::<u32>()?;
        let num_grid = self.num_grid_per_side();
        let hidden_size = self.config.embedding_length;
        let merge_size = self.config.spatial_merge_size;

        let mut all_embeds = Vec::new();
        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;

            // Compute interpolation coordinates and weights using tensor ops
            // linspace: generate interpolation coordinates
            let max_val = (num_grid - 1) as f32;
            let h_step = if h > 1 { max_val / (h - 1) as f32 } else { 0.0 };
            let w_step = if w > 1 { max_val / (w - 1) as f32 } else { 0.0 };

            // Build indices and weights for all h*w positions at once
            let hw = h * w;
            let mut idx00_vec = Vec::with_capacity(hw);
            let mut idx01_vec = Vec::with_capacity(hw);
            let mut idx10_vec = Vec::with_capacity(hw);
            let mut idx11_vec = Vec::with_capacity(hw);
            let mut w00_vec = Vec::with_capacity(hw);
            let mut w01_vec = Vec::with_capacity(hw);
            let mut w10_vec = Vec::with_capacity(hw);
            let mut w11_vec = Vec::with_capacity(hw);

            for hi in 0..h {
                let hv = hi as f32 * h_step;
                let h_floor = hv.floor() as usize;
                let h_ceil = (hv.ceil() as usize).min(num_grid - 1);
                let dh = hv - h_floor as f32;

                for wi in 0..w {
                    let wv = wi as f32 * w_step;
                    let w_floor = wv.floor() as usize;
                    let w_ceil = (wv.ceil() as usize).min(num_grid - 1);
                    let dw = wv - w_floor as f32;

                    idx00_vec.push((h_floor * num_grid + w_floor) as u32);
                    idx01_vec.push((h_floor * num_grid + w_ceil) as u32);
                    idx10_vec.push((h_ceil * num_grid + w_floor) as u32);
                    idx11_vec.push((h_ceil * num_grid + w_ceil) as u32);

                    w00_vec.push((1.0 - dh) * (1.0 - dw));
                    w01_vec.push((1.0 - dh) * dw);
                    w10_vec.push(dh * (1.0 - dw));
                    w11_vec.push(dh * dw);
                }
            }

            // Create index tensors and gather all corners in batch
            let idx00 = Tensor::from_vec(idx00_vec, hw, device)?;
            let idx01 = Tensor::from_vec(idx01_vec, hw, device)?;
            let idx10 = Tensor::from_vec(idx10_vec, hw, device)?;
            let idx11 = Tensor::from_vec(idx11_vec, hw, device)?;

            // Batch gather: [hw, hidden_size] for each corner
            let p00 = self.position_embed.index_select(&idx00, 0)?;
            let p01 = self.position_embed.index_select(&idx01, 0)?;
            let p10 = self.position_embed.index_select(&idx10, 0)?;
            let p11 = self.position_embed.index_select(&idx11, 0)?;

            // Create weight tensors and broadcast multiply
            let w00 = Tensor::from_vec(w00_vec, (hw, 1), device)?.to_dtype(dtype)?;
            let w01 = Tensor::from_vec(w01_vec, (hw, 1), device)?.to_dtype(dtype)?;
            let w10 = Tensor::from_vec(w10_vec, (hw, 1), device)?.to_dtype(dtype)?;
            let w11 = Tensor::from_vec(w11_vec, (hw, 1), device)?.to_dtype(dtype)?;

            // Weighted sum: [hw, hidden_size]
            let pos_embed_hw = ((p00.broadcast_mul(&w00)?
                + p01.broadcast_mul(&w01)?)?
                + p10.broadcast_mul(&w10)?)?
                .broadcast_add(&p11.broadcast_mul(&w11)?)?;

            // Repeat for temporal dimension: [hw, hidden] -> [t*hw, hidden]
            let pos_embed_thw = pos_embed_hw.repeat((t, 1))?;

            // Permute for spatial merge: reorder to interleave merge blocks
            // [t*h*w, hidden] -> [t, h/m, m, w/m, m, hidden] -> permute -> [t, h/m, w/m, m, m, hidden] -> [t*h*w, hidden]
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
                .contiguous()?  // Make contiguous after permute for efficient reshape
                .reshape((t * h * w, hidden_size))?;
            all_embeds.push(pos_embed);
        }

        Tensor::cat(&all_embeds, 0)?.to_dtype(dtype)
    }

    /// Apply patch embedding using two Conv2D weights (from GGUF mmproj)
    ///
    /// GGUF mmproj splits the Conv3D kernel into two Conv2D kernels:
    /// - v.patch_embd.weight: temporal slice 0
    /// - v.patch_embd.weight.1: temporal slice 1
    ///
    /// The correct operation is: output = xs_t0 @ W0.T + xs_t1 @ W1.T + bias
    /// where xs_t0/xs_t1 are the two temporal slices of the input.
    fn patch_embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let debug = std::env::var("MISTRALRS_DEBUG_VISION").is_ok();

        // xs shape: [num_patches, C*T*H*W] where C=3, T=2, H=W=16
        // Input features are ordered as [channel, temporal, h, w] from preprocessing
        let dtype = xs.dtype();

        if debug {
            let xs_f32 = xs.to_dtype(candle_core::DType::F32)?;
            let xs_mean = xs_f32.mean_all()?.to_scalar::<f32>()?;
            let xs_var = xs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] patch_embed_forward INPUT:");
            eprintln!("  xs shape: {:?}, dtype: {:?}", xs.dims(), xs.dtype());
            eprintln!("  xs mean: {:.6}, std: {:.6}", xs_mean, xs_var.sqrt());
            // Print first few values
            let first_vals: Vec<f32> = xs_f32.flatten_all()?.narrow(0, 0, 10.min(xs_f32.elem_count()))?.to_vec1()?;
            eprintln!("  xs first 10 values: {:?}", first_vals);
        }

        // Weight shape: [out_channels, in_channels, H, W] = [1024, 3, 16, 16]
        let weight_shape = self.patch_embed_weight_0.dims();
        let out_channels = weight_shape[0];
        let in_channels = weight_shape[1];
        let patch_h = weight_shape[2];
        let patch_w = weight_shape[3];
        let hw = patch_h * patch_w;  // 256
        let in_features = in_channels * hw;  // 768 = 3 * 16 * 16

        if debug {
            eprintln!("  weight_0 shape: {:?}", self.patch_embed_weight_0.dims());
            eprintln!("  weight_1 shape: {:?}", self.patch_embed_weight_1.dims());
            eprintln!("  out_channels={}, in_channels={}, hw={}, in_features={}", out_channels, in_channels, hw, in_features);
        }

        // Convert weights and bias to input dtype
        // Use contiguous() to ensure GGUF-loaded weights have proper memory layout
        let weight_0 = self.patch_embed_weight_0.to_dtype(dtype)?.contiguous()?;
        let weight_1 = self.patch_embed_weight_1.to_dtype(dtype)?.contiguous()?;
        let bias = self.patch_embed_bias.to_dtype(dtype)?;

        if debug {
            let w0_f32 = weight_0.to_dtype(candle_core::DType::F32)?;
            let w1_f32 = weight_1.to_dtype(candle_core::DType::F32)?;
            let w0_mean = w0_f32.mean_all()?.to_scalar::<f32>()?;
            let w1_mean = w1_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!("  weight_0 mean: {:.6}, weight_1 mean: {:.6}", w0_mean, w1_mean);
        }

        // Flatten weights: [O, C, H, W] -> [O, C*H*W]
        let w0_flat = weight_0.reshape((out_channels, in_features))?;
        let w1_flat = weight_1.reshape((out_channels, in_features))?;

        // Input shape
        let num_patches = xs.dims()[0];
        let total_features = xs.dims()[1];
        let temporal = 2usize;
        let expected_features = in_channels * temporal * hw;  // 3 * 2 * 256 = 1536

        if debug {
            eprintln!("  num_patches={}, total_features={}, expected={}", num_patches, total_features, expected_features);
        }

        // Validate input shape
        if total_features != expected_features {
            if debug {
                eprintln!("  WARNING: total_features ({}) != expected ({})", total_features, expected_features);
            }
        }

        // Reshape input: [num_patches, C*T*H*W] -> [num_patches, C, T, H*W]
        // HuggingFace Conv3d expects [N, C, T, H, W] layout, so features should be in [C, T, H, W] order
        // This matches HuggingFace's reshape(((), in_channels, temporal_patch_size, patch_size, patch_size))
        let xs_reshaped = xs.reshape((num_patches, in_channels, temporal, hw))?;

        // Extract temporal slices: [num_patches, C, T, H*W] -> select T dim -> [num_patches, C, H*W] -> flatten to [num_patches, C*H*W]
        // Use contiguous() to ensure proper memory layout before reshape
        let xs_t0 = xs_reshaped.i((.., .., 0, ..))?.contiguous()?.reshape((num_patches, in_features))?;
        let xs_t1 = xs_reshaped.i((.., .., 1, ..))?.contiguous()?.reshape((num_patches, in_features))?;

        if debug {
            let t0_f32 = xs_t0.to_dtype(candle_core::DType::F32)?;
            let t1_f32 = xs_t1.to_dtype(candle_core::DType::F32)?;
            let t0_mean = t0_f32.mean_all()?.to_scalar::<f32>()?;
            let t1_mean = t1_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!("  xs_t0 shape: {:?}, mean: {:.6}", xs_t0.dims(), t0_mean);
            eprintln!("  xs_t1 shape: {:?}, mean: {:.6}", xs_t1.dims(), t1_mean);

            // Verify: for [T, C, H*W] layout, T=0 slice contains all 3 channels
            // Original layout: [T0C0, T0C1, T0C2, T1C0, T1C1, T1C2] where each segment is 256 (H*W)
            // xs_t0 should have: R[0:256], G[256:512], B[512:768] (all from temporal frame 0)
            let xs_orig_f32 = xs.to_dtype(candle_core::DType::F32)?;
            let orig_first = xs_orig_f32.i(0)?.to_vec1::<f32>()?;
            let t0_first = t0_f32.i(0)?.to_vec1::<f32>()?;

            // Check if T=0 slice extracts correct positions from original
            // With [T, C, H*W] layout: T0C0 is [0:256], T0C1 is [256:512], T0C2 is [512:768]
            //                          T1C0 is [768:1024], T1C1 is [1024:1280], T1C2 is [1280:1536]
            eprintln!("  Verification - Original vs T0 extraction:");
            eprintln!("    orig[0] (T0C0[0])   = {:.6}, t0[0] (should be T0C0[0])   = {:.6}", orig_first[0], t0_first[0]);
            eprintln!("    orig[256] (T0C1[0]) = {:.6}, t0[256] (should be T0C1[0]) = {:.6}", orig_first[256], t0_first[256]);
            eprintln!("    orig[512] (T0C2[0]) = {:.6}, t0[512] (should be T0C2[0]) = {:.6}", orig_first[512], t0_first[512]);
            // T1C0[0] should be at orig position 768, which is NOT in t0 (t0 only has T=0 data)
            eprintln!("    orig[768] (T1C0[0]) = {:.6} (should NOT appear in t0)", orig_first[768]);

            // Show first patch's values to understand layout
            // xs_t0 has shape [num_patches, C*H*W] = [num_patches, 768]
            let patch0 = t0_f32.i(0)?.to_vec1::<f32>()?;

            // Hypothesis 1: Contiguous channels [R_all, G_all, B_all]
            let r_mean_cont: f32 = patch0[0..256].iter().sum::<f32>() / 256.0;
            let g_mean_cont: f32 = patch0[256..512].iter().sum::<f32>() / 256.0;
            let b_mean_cont: f32 = patch0[512..768].iter().sum::<f32>() / 256.0;

            // Hypothesis 2: Interleaved [R0,G0,B0,R1,G1,B1,...]
            let mut r_sum = 0.0f32;
            let mut g_sum = 0.0f32;
            let mut b_sum = 0.0f32;
            for i in 0..256 {
                r_sum += patch0[i * 3];
                g_sum += patch0[i * 3 + 1];
                b_sum += patch0[i * 3 + 2];
            }
            let r_mean_int = r_sum / 256.0;
            let g_mean_int = g_sum / 256.0;
            let b_mean_int = b_sum / 256.0;

            eprintln!("  Patch 0 (CONTIGUOUS [C,H*W]): R={:.3}, G={:.3}, B={:.3}", r_mean_cont, g_mean_cont, b_mean_cont);
            eprintln!("  Patch 0 (INTERLEAVED [RGB,RGB,...]): R={:.3}, G={:.3}, B={:.3}", r_mean_int, g_mean_int, b_mean_int);

            // Hypothesis 3: Layout is [H*W, C] not [C, H*W] - so 768 = 256 pixels * 3 channels
            // Each position i has RGB triplet: patch0[i*3], patch0[i*3+1], patch0[i*3+2]
            // Wait, that's same as interleaved. Let me try [H, W, C] style
            // If 768 = 16*16*3, then groups of 3 are RGB values for each pixel
            // First pixel: patch0[0..3], second pixel: patch0[3..6], etc.
            // This is same as interleaved interpretation above.

            // Let's check: are the first 256 values all unique or mostly same?
            let unique_in_first_256: std::collections::HashSet<u32> = patch0[0..256]
                .iter()
                .map(|x| x.to_bits())
                .collect();
            eprintln!("  Unique values in first 256: {}", unique_in_first_256.len());

            // Print every 256th value to see the pattern
            eprintln!("  Values at pos 0, 256, 512: [{:.3}, {:.3}, {:.3}]", patch0[0], patch0[256], patch0[512]);
            eprintln!("  Values at pos 1, 257, 513: [{:.3}, {:.3}, {:.3}]", patch0[1], patch0[257], patch0[513]);

            eprintln!("  First 12 values: {:?}", &patch0[0..12]);
        }

        // Apply different weights to different temporal slices
        // This matches Conv3D behavior: output = xs_t0 @ W0.T + xs_t1 @ W1.T
        let out_t0 = xs_t0.matmul(&w0_flat.t()?)?;
        let out_t1 = xs_t1.matmul(&w1_flat.t()?)?;

        if debug {
            let out0_f32 = out_t0.to_dtype(candle_core::DType::F32)?;
            let out1_f32 = out_t1.to_dtype(candle_core::DType::F32)?;
            let out0_mean = out0_f32.mean_all()?.to_scalar::<f32>()?;
            let out1_mean = out1_f32.mean_all()?.to_scalar::<f32>()?;
            eprintln!("  out_t0 shape: {:?}, mean: {:.6}", out_t0.dims(), out0_mean);
            eprintln!("  out_t1 shape: {:?}, mean: {:.6}", out_t1.dims(), out1_mean);
        }

        // Sum and add bias
        let out = out_t0.add(&out_t1)?.broadcast_add(&bias)?;

        if debug {
            let out_f32 = out.to_dtype(candle_core::DType::F32)?;
            let out_mean = out_f32.mean_all()?.to_scalar::<f32>()?;
            let out_var = out_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] patch_embed_forward OUTPUT:");
            eprintln!("  out shape: {:?}, mean: {:.6}, std: {:.6}", out.dims(), out_mean, out_var.sqrt());
        }

        Ok(out)
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
        let device = self.position_embed.device();
        let debug = std::env::var("MISTRALRS_DEBUG_VISION").is_ok();

        if debug {
            eprintln!("\n[DEBUG] ========== VISION ENCODER FORWARD ==========");
            eprintln!("  pixel_values shape: {:?}, dtype: {:?}", pixel_values.dims(), pixel_values.dtype());
            eprintln!("  grid_thw shape: {:?}", grid_thw.dims());
            let grid_vals: Vec<u32> = grid_thw.to_dtype(candle_core::DType::U32)?.flatten_all()?.to_vec1()?;
            eprintln!("  grid_thw values: {:?}", grid_vals);
        }

        // Timing instrumentation (enable via MISTRALRS_PROFILE_VISION=1)
        let profile = std::env::var("MISTRALRS_PROFILE_VISION").is_ok();
        let mut timings: Vec<(&str, std::time::Duration)> = Vec::new();
        let total_start = std::time::Instant::now();

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
        let t0 = std::time::Instant::now();
        let xs = self.patch_embed_forward(&pixel_values.to_dtype(dtype)?)?;
        if profile { device.synchronize()?; timings.push(("patch_embed", t0.elapsed())); }

        if debug {
            let xs_f32 = xs.to_dtype(candle_core::DType::F32)?;
            let xs_mean = xs_f32.mean_all()?.to_scalar::<f32>()?;
            let xs_var = xs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] After patch_embed: shape={:?}, mean={:.6}, std={:.6}", xs.dims(), xs_mean, xs_var.sqrt());
        }

        // 2. Position embedding
        let t0 = std::time::Instant::now();
        let pos_embeds = self.interpolate_pos_embed(grid_thw)?;
        let mut hidden_states = xs.add(&pos_embeds)?;
        if profile { device.synchronize()?; timings.push(("pos_embed", t0.elapsed())); }

        if debug {
            let hs_f32 = hidden_states.to_dtype(candle_core::DType::F32)?;
            let hs_mean = hs_f32.mean_all()?.to_scalar::<f32>()?;
            let hs_var = hs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] After pos_embed: shape={:?}, mean={:.6}, std={:.6}", hidden_states.dims(), hs_mean, hs_var.sqrt());
        }

        // 3. Rotary position embedding for attention
        // OPTIMIZED: Keep cos/sin in native dtype - Metal supports F16/BF16 for all RoPE ops
        let t0 = std::time::Instant::now();
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let head_dim = self.config.embedding_length / self.config.num_heads;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, head_dim))?;
        // cos/sin shape should be [seq_len, head_dim] to match q/k [seq_len, num_heads, head_dim]
        // Keep in hidden_states dtype (F16) - no F32 conversion needed
        let cos = rotary_pos_emb.cos()?.to_dtype(hidden_states.dtype())?;
        let sin = rotary_pos_emb.sin()?.to_dtype(hidden_states.dtype())?;
        if profile { device.synchronize()?; timings.push(("rope_coords", t0.elapsed())); }

        // 4. Build cumulative sequence lengths
        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        // 5. Process through transformer blocks with deepstack
        let t0 = std::time::Instant::now();
        let debug_blocks = std::env::var("MISTRALRS_DEBUG_VISION_BLOCKS").is_ok();
        let mut deepstack_features = Vec::new();
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            if debug_blocks && layer_idx == 0 {
                hidden_states = block.forward_debug(&hidden_states, &cu_seqlens, &cos, &sin, layer_idx)?;
            } else {
                hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
            }

            // Check if this layer has a deepstack merger
            if let Some(deepstack_layer) = self.deepstack.get(&layer_idx) {
                let feat = deepstack_layer.forward(&hidden_states)?;
                deepstack_features.push(feat);
            }
        }
        if profile { device.synchronize()?; timings.push(("transformer_blocks", t0.elapsed())); }

        if debug {
            let hs_f32 = hidden_states.to_dtype(candle_core::DType::F32)?;
            let hs_mean = hs_f32.mean_all()?.to_scalar::<f32>()?;
            let hs_var = hs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] After transformer_blocks: shape={:?}, mean={:.6}, std={:.6}", hidden_states.dims(), hs_mean, hs_var.sqrt());
        }

        // 6. Post layer norm
        let t0 = std::time::Instant::now();
        hidden_states = self.post_ln.forward(&hidden_states)?;
        if profile { device.synchronize()?; timings.push(("post_ln", t0.elapsed())); }

        if debug {
            let hs_f32 = hidden_states.to_dtype(candle_core::DType::F32)?;
            let hs_mean = hs_f32.mean_all()?.to_scalar::<f32>()?;
            let hs_var = hs_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] After post_ln: shape={:?}, mean={:.6}, std={:.6}", hidden_states.dims(), hs_mean, hs_var.sqrt());
        }

        // Print timing summary
        if profile {
            let total = total_start.elapsed();
            tracing::info!("Vision encoder timing breakdown ({} tokens):", seq_len);
            for (name, dur) in &timings {
                let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
                tracing::info!("  {:<20} {:>8.2}ms ({:>5.1}%)", name, dur.as_secs_f64() * 1000.0, pct);
            }
            tracing::info!("  {:<20} {:>8.2}ms", "TOTAL", total.as_secs_f64() * 1000.0);
        }

        // 7. Merger (projects to output dimension)
        let merged = self.merger.forward(&hidden_states)?;

        if debug {
            let m_f32 = merged.to_dtype(candle_core::DType::F32)?;
            let m_mean = m_f32.mean_all()?.to_scalar::<f32>()?;
            let m_var = m_f32.var(0)?.mean_all()?.to_scalar::<f32>()?;
            eprintln!("[DEBUG] After merger (FINAL): shape={:?}, mean={:.6}, std={:.6}", merged.dims(), m_mean, m_var.sqrt());
            eprintln!("[DEBUG] ========== END VISION ENCODER ==========\n");
        }

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
