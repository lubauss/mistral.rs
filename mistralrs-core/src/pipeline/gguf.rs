use super::llg::build_llg_factory;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, PrettyName, Processor,
    QuantizationKind, TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, IsqPipelineMixin,
    MetadataMixin, ModelCategory, MultimodalPromptPrefixer, PreProcessingMixin,
};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::qwen3_vl::{inputs_processor::Qwen3VLProcessor, Qwen3VLVisionSpecificArgs};
use crate::vision_models::ModelInputs as VisionModelInputs;
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::gguf::{
    get_gguf_chat_template, {convert_gguf_to_hf_tokenizer, GgufTokenizerConversion},
};
use crate::gguf::{Content, GGUFArchitecture};
use crate::kv_cache::{FullCacheManager, NormalCacheManager};
use crate::lora::Ordering;
use crate::paged_attention::{
    calculate_cache_config, AttentionImplementation, CacheEngine, ModelConfigLike,
};
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
use crate::pipeline::loaders::DeviceMappedModelLoader;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::gguf_metadata::{ContentConfig, GgufDeviceMapLoaderInner};
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::ProgressScopeGuard;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    get_mut_arcmutex, get_paths_gguf, DeviceMapSetting, LocalModelPaths, PagedAttentionConfig,
    Pipeline, Topology, TryIntoDType,
};
use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    models::quantized_qwen::ModelWeights as QQwen,
    models::quantized_qwen3::ModelWeights as QQwen3,
    models::quantized_qwen3_moe::ModelWeights as QQwen3MoE,
    models::quantized_qwen3_vl::ModelWeights as QQwen3Vl,
    models::quantized_qwen3_vl_moe::ModelWeights as QQwen3VlMoE,
    models::quantized_qwen3_vl_vision::QQwen3VLVisionEncoder,
    models::quantized_starcoder2::ModelWeights as QStarcoder2,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use either::Either;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

enum Model {
    Llama(QLlama),
    Phi2(QPhi),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
    Phi3(QPhi3),
    Starcoder2(QStarcoder2),
    Qwen(QQwen),
    Qwen3(QQwen3),
    Qwen3MoE(QQwen3MoE),
    Qwen3Vl(QQwen3Vl),
    Qwen3VlMoE(QQwen3VlMoE),
}

impl Model {
    fn is_vision(&self) -> bool {
        matches!(self, Model::Qwen3Vl(_) | Model::Qwen3VlMoE(_))
    }
}

/// No-op prefixer for GGUF vision models.
/// The chat template handles image tokens via MessagesAction::Keep.
struct GGUFVisionPrefixer;

impl MultimodalPromptPrefixer for GGUFVisionPrefixer {}

pub struct GGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    /// Vision-specific: processor for image preprocessing
    processor: Option<Arc<dyn Processor + Send + Sync>>,
    /// Vision-specific: preprocessor config for image parameters
    preprocessor_config: Option<Arc<PreProcessorConfig>>,
}

/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
    lora_adapter_ids: Option<Vec<String>>,
}

#[derive(Clone, Default)]
/// Config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub topology: Option<Topology>,
    /// Path to multimodal projector GGUF file for vision models.
    /// This is required for GGUF vision models like Qwen3-VL.
    pub mmproj_path: Option<String>,
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tgt_non_granular_index: Option<usize>,
    config: GGUFSpecificConfig,
    jinja_explicit: Option<String>,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` is the model ID where you can find a
    /// `tokenizer_config.json` file. If the `chat_template` is specified, then it will be treated as a
    /// path and used over remote files, removing all remote accesses.
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: GGUFSpecificConfig,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let kind = ModelKind::GgufQuantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            chat_template,
            model_id: tok_model_id,
            kind,
            quantized_filenames,
            quantized_model_id,
            config,
            jinja_explicit,
            no_kv_cache,
            ..Default::default()
        }
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = (AdapterKind::XLora, QuantizationKind::Gguf).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        Box::new(GGUFLoader {
            model_id: self.model_id,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filenames: self.quantized_filenames,
            quantized_model_id: self.quantized_model_id,
            config: self.config,
            jinja_explicit: self.jinja_explicit,
            lora_adapter_ids: None,
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
        config: GGUFSpecificConfig,
        jinja_explicit: Option<String>,
    ) -> Self {
        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
        };
        Self {
            model_id,
            quantized_model_id,
            quantized_filenames,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            kind,
            tgt_non_granular_index,
            config,
            jinja_explicit,
            lora_adapter_ids: None,
        }
    }
}

impl Loader for GGUFLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id.clone(),
            self.quantized_filenames.clone(),
            silent
        );
        self.load_model_from_path(
            &paths?,
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }

        info!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let mut readers = Vec::new();
        for filename in paths.get_weight_filenames() {
            readers.push(std::fs::File::open(filename)?);
        }
        let mut readers = readers.iter_mut().collect::<Vec<_>>();

        let model = Content::from_readers(&mut readers)?;
        if !silent {
            model.print_metadata()?;
        }
        let arch = model.arch();

        // Validate mmproj configuration
        if arch.supports_vision() && self.config.mmproj_path.is_none() {
            anyhow::bail!(
                "Vision architecture `{arch:?}` requires --mmproj flag to specify the multimodal projector GGUF file"
            );
        }
        if !arch.supports_vision() && self.config.mmproj_path.is_some() {
            tracing::warn!(
                "--mmproj flag will be ignored for non-vision architecture `{arch:?}`"
            );
        }

        // If auto, convert to Map
        let num_layers = model.get_metadata()[&format!("{arch}.block_count")].to_u32()? as usize;
        if let DeviceMapSetting::Auto(params) = mapper.clone() {
            let devices = device_map::get_all_similar_devices(device)?;
            // Initial dtype
            let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;

            let model = GgufDeviceMapLoaderInner {
                model: &model,
                arch,
            };

            let layer_sizes_in_bytes =
                model.layer_sizes_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let non_mapped_size_in_bytes =
                model.non_mapped_size_in_bytes("this is a dummy config!", dtype, 1, None)?;
            let total_model_size_in_bytes =
                layer_sizes_in_bytes.iter().sum::<usize>() + non_mapped_size_in_bytes;

            let new = model.get_device_layers(
                "this is a dummy config!",
                num_layers,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &devices,
                dtype,
                &params,
                paged_attn_config.as_ref(),
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }

        let pipeline_mapper =
            mapper.into_mapper(num_layers, device, self.config.topology.as_ref())?;
        let mapper = mapper.into_mapper(num_layers, device, self.config.topology.as_ref())?;
        let mut layer_devices = Vec::new();
        for layer in 0..num_layers {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
            convert_gguf_to_hf_tokenizer(&model)?
        } else {
            GgufTokenizerConversion {
                tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
                bos: None,
                eos: None,
                unk: None,
            }
        };

        // Only load gguf chat template if there is nothing else
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template(&model)?
            } else {
                None
            };

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let paged_attn_config = if matches!(self.kind, ModelKind::GgufAdapter { .. }) {
            warn!("Adapter models do not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let model_config_metadata: ContentConfig = (&model).into();
        let internal_dtype = mapper.get_min_dtype(dtype)?;

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGUF(
                model,
                (device, mapper).into(),
                if paged_attn_config.is_some() {
                    AttentionImplementation::PagedAttention
                } else {
                    AttentionImplementation::Eager
                },
                internal_dtype,
            );

            // With optional adapter config:
            let mut adapter = None;
            if has_adapter {
                adapter.replace(ModelConfig::Adapter::try_new(
                    paths, device, silent, is_xlora,
                )?);
            }

            ModelConfig::ModelParams::new(quant, adapter)
        };

        // Config into model:
        let model = match self.kind {
            ModelKind::GgufQuantized { .. } => match arch {
                GGUFArchitecture::Llama => Model::Llama(QLlama::try_from(model_config)?),
                GGUFArchitecture::Phi2 => Model::Phi2(QPhi::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::Phi3(QPhi3::try_from(model_config)?),
                GGUFArchitecture::Starcoder2 => {
                    Model::Starcoder2(QStarcoder2::try_from(model_config)?)
                }
                GGUFArchitecture::Qwen2 => Model::Qwen(QQwen::try_from(model_config)?),
                GGUFArchitecture::Qwen3 => Model::Qwen3(QQwen3::try_from(model_config)?),
                GGUFArchitecture::Qwen3MoE => Model::Qwen3MoE(QQwen3MoE::try_from(model_config)?),
                GGUFArchitecture::Qwen3Vl | GGUFArchitecture::Qwen3VlMoE => {
                    // Vision architecture requires loading mmproj file separately
                    let mmproj_path = self.config.mmproj_path.as_ref()
                        .expect("mmproj_path should be validated earlier");

                    info!("Loading multimodal projector from: {mmproj_path}");

                    // Load mmproj GGUF content (separate from LLM content)
                    let mut mmproj_file = std::fs::File::open(mmproj_path)
                        .map_err(|e| anyhow::anyhow!(
                            "Failed to open mmproj file '{}': {e}",
                            mmproj_path
                        ))?;
                    let mut mmproj_readers: Vec<&mut std::fs::File> = vec![&mut mmproj_file];
                    let mut mmproj_content = Content::from_readers(&mut mmproj_readers)?;

                    // Validate mmproj architecture
                    let mmproj_arch = mmproj_content.arch();
                    info!("mmproj architecture: {mmproj_arch:?}");
                    if !mmproj_arch.is_vision_encoder() {
                        bail!(
                            "mmproj file has unexpected architecture `{mmproj_arch:?}`, expected CLIP. \
                            Ensure you're providing a vision encoder mmproj file, not an LLM GGUF."
                        );
                    }

                    // Load vision encoder from mmproj content
                    let vision_encoder = QQwen3VLVisionEncoder::load(&mut mmproj_content, device)?;
                    let vision_config = vision_encoder.config();
                    info!(
                        "Loaded vision encoder: {} blocks, {} embed dim, {} spatial merge",
                        vision_config.block_count,
                        vision_config.embedding_length,
                        vision_config.spatial_merge_size
                    );

                    // Load LLM text model - use MoE variant for qwen3vlmoe
                    match arch {
                        GGUFArchitecture::Qwen3VlMoE => {
                            let mut llm_model = QQwen3VlMoE::try_from(model_config)?;
                            llm_model.set_vision_encoder(vision_encoder);
                            info!("Vision encoder attached to MoE LLM model");
                            Model::Qwen3VlMoE(llm_model)
                        }
                        _ => {
                            let mut llm_model = QQwen3Vl::try_from(model_config)?;
                            llm_model.set_vision_encoder(vision_encoder);
                            info!("Vision encoder attached to LLM model");
                            Model::Qwen3Vl(llm_model)
                        }
                    }
                }
                GGUFArchitecture::Clip => {
                    bail!("CLIP architecture is for mmproj files only, not standalone models. \
                          Use --mmproj flag with a vision-capable LLM architecture like qwen3vl.");
                }
                a => bail!("Unsupported architecture `{a:?}` for GGUF"),
            },
            ModelKind::GgufAdapter { adapter, .. } => match arch {
                GGUFArchitecture::Llama => Model::XLoraLlama(XLoraQLlama::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::try_from(model_config)?),
                a => bail!(
                    "Unsupported architecture `{a:?}` for GGUF {kind}",
                    kind = adapter.pretty_name()
                ),
            },
            _ => unreachable!(),
        };

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            let model_config: &dyn ModelConfigLike = &model_config_metadata;
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.block_size,
                internal_dtype,
                paged_attn_config.cache_type,
                model_config,
                device,
                &layer_devices,
                silent,
            )?;
            let cache_engine = CacheEngine::new(
                model_config,
                &cache_config,
                internal_dtype,
                device,
                layer_devices,
            )?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let mut chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            gguf_chat_template,
        );

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::Phi2(ref p) => p.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
            Model::Phi3(ref p) => p.max_seq_len,
            Model::XLoraPhi3(ref p) => p.max_seq_len,
            Model::Starcoder2(ref p) => p.max_seq_len,
            Model::Qwen(ref p) => p.max_seq_len,
            Model::Qwen3(ref p) => p.max_seq_len,
            Model::Qwen3MoE(ref p) => p.max_seq_len,
            Model::Qwen3Vl(ref p) => p.max_seq_len,
            Model::Qwen3VlMoE(ref p) => p.max_seq_len,
        };
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.normal().0.len(),
            Model::Phi2(ref model) => model.cache.normal().0.len(),
            Model::XLoraLlama(ref model) => model.cache.full().lock().len(),
            Model::Phi3(ref model) => model.cache.normal().0.len(),
            Model::XLoraPhi3(ref model) => model.cache.full().lock().len(),
            Model::Starcoder2(ref model) => model.cache.normal().0.len(),
            Model::Qwen(ref model) => model.cache.normal().0.len(),
            Model::Qwen3(ref model) => model.cache.normal().0.len(),
            Model::Qwen3MoE(ref model) => model.cache.normal().0.len(),
            Model::Qwen3Vl(ref model) => model.cache.normal().0.len(),
            Model::Qwen3VlMoE(ref model) => model.cache.normal().0.len(),
        };

        if chat_template.bos_token.is_none() {
            if let Some(v) = bos {
                chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.eos_token.is_none() {
            if let Some(v) = eos {
                chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }
        if chat_template.unk_token.is_none() {
            if let Some(v) = unk {
                chat_template.unk_token = Some(BeginEndUnkPadTok(Either::Left(v)));
            }
        }

        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        let is_vision_model = model.is_vision();
        // Set up vision processor and config for vision models
        let (processor, preprocessor_config): (
            Option<Arc<dyn Processor + Send + Sync>>,
            Option<Arc<PreProcessorConfig>>,
        ) = if is_vision_model {
            // Create preprocessor config matching GGUF mmproj parameters
            // GGUF Qwen3-VL uses patch_size=16, temporal_patch_size=2, merge_size=2
            let mut config = PreProcessorConfig::default();
            config.patch_size = Some(16);  // GGUF mmproj uses 16, not 14
            config.temporal_patch_size = Some(2);
            config.merge_size = Some(2);
            (
                Some(Arc::new(Qwen3VLProcessor::new(None))),
                Some(Arc::new(config)),
            )
        } else {
            (None, None)
        };

        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self
                .model_id
                .clone()
                .unwrap_or(self.quantized_model_id.clone()),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                no_kv_cache: self.no_kv_cache,
                no_prefix_cache: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: internal_dtype,
                sliding_window: None,
                cache_config,
                cache_engine,
                model_metadata: Some(Arc::new(model_config_metadata)),
                modalities: Modalities {
                    input: if is_vision_model {
                        vec![SupportedModality::Text, SupportedModality::Vision]
                    } else {
                        vec![SupportedModality::Text]
                    },
                    output: vec![SupportedModality::Text],
                },
            }),
            mapper: pipeline_mapper,
            processor,
            preprocessor_config,
        })))
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGUFPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        self.preprocessor_config
            .as_ref()
            .map(|c| c.clone() as Arc<dyn Any>)
    }
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor
            .clone()
            .unwrap_or_else(|| Arc::new(super::processing::BasicProcessor))
    }
}

impl IsqPipelineMixin for GGUFPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGUFPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_in_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_in_cache(self, seqs, false)
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_out_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_out_cache(self, seqs, false)
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) {
        if matches!(self.cache(), EitherCache::Full(_)) {
            FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false);
        } else {
            NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            );
        }
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::Phi2(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
            Model::Phi3(ref model) => &model.cache,
            Model::XLoraPhi3(ref model) => &model.cache,
            Model::Starcoder2(ref model) => &model.cache,
            Model::Qwen(ref model) => &model.cache,
            Model::Qwen3(ref model) => &model.cache,
            Model::Qwen3MoE(ref model) => &model.cache,
            Model::Qwen3Vl(ref model) => &model.cache,
            Model::Qwen3VlMoE(ref model) => &model.cache,
        }
    }
}

impl MetadataMixin for GGUFPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::Phi2(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
            Model::Phi3(ref model) => model.device.clone(),
            Model::XLoraPhi3(ref model) => model.device.clone(),
            Model::Starcoder2(ref model) => model.device.clone(),
            Model::Qwen(ref model) => model.device.clone(),
            Model::Qwen3(ref model) => model.device.clone(),
            Model::Qwen3MoE(ref model) => model.device.clone(),
            Model::Qwen3Vl(ref model) => model.device.clone(),
            Model::Qwen3VlMoE(ref model) => model.device.clone(),
        }
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for GGUFPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        let metadata = self.get_metadata();

        // Try to downcast to vision inputs first for vision models
        let logits = if self.model.is_vision() {
            // Try vision model inputs first (has pixel_values and model_specific_args)
            if let Ok(vision_inputs) = inputs.downcast::<VisionModelInputs>() {
                let VisionModelInputs {
                    input_ids,
                    seqlen_offsets,
                    context_lens,
                    position_ids: _,
                    pixel_values,
                    model_specific_args,
                    paged_attn_meta,
                    flash_meta: _,
                } = *vision_inputs;

                let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
                    (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
                    (Some(_), None) => {
                        candle_core::bail!("Forward step expected a PagedAttention input metadata.")
                    }
                    (None, Some(_)) => {
                        candle_core::bail!("Forward step got PagedAttention metadata but no cache engine.")
                    }
                    (None, None) => None,
                };

                match &self.model {
                    Model::Qwen3Vl(model) => {
                        // Extract vision-specific args
                        let Qwen3VLVisionSpecificArgs {
                            input_ids_full: _,
                            image_grid_thw,
                            video_grid_thw: _,
                            seqlens: _,
                            continuous_img_pad,
                            continuous_vid_pad: _,
                        } = *model_specific_args
                            .downcast()
                            .expect("Cannot downcast into Qwen3VLVisionSpecificArgs");

                        // Build image_positions from continuous_img_pad
                        // Format: (batch_idx, start_pos, end_pos)
                        let mut image_positions = Vec::new();
                        for (batch_idx, pads) in continuous_img_pad.iter().enumerate() {
                            for (start, end) in pads {
                                image_positions.push((batch_idx, *start, *end));
                            }
                        }

                        model.forward_with_vision(
                            &input_ids,
                            pixel_values.as_ref(),
                            image_grid_thw.as_ref(),
                            &image_positions,
                            &seqlen_offsets,
                            context_lens,
                            paged_attn_meta,
                        )?
                    }
                    Model::Qwen3VlMoE(model) => {
                        // Extract vision-specific args
                        let Qwen3VLVisionSpecificArgs {
                            input_ids_full: _,
                            image_grid_thw,
                            video_grid_thw: _,
                            seqlens: _,
                            continuous_img_pad,
                            continuous_vid_pad: _,
                        } = *model_specific_args
                            .downcast()
                            .expect("Cannot downcast into Qwen3VLVisionSpecificArgs");

                        // Build image_positions from continuous_img_pad
                        let mut image_positions = Vec::new();
                        for (batch_idx, pads) in continuous_img_pad.iter().enumerate() {
                            for (start, end) in pads {
                                image_positions.push((batch_idx, *start, *end));
                            }
                        }

                        model.forward_with_vision(
                            &input_ids,
                            pixel_values.as_ref(),
                            image_grid_thw.as_ref(),
                            &image_positions,
                            &seqlen_offsets,
                            context_lens,
                            paged_attn_meta,
                        )?
                    }
                    _ => {
                        candle_core::bail!("Vision inputs received for non-vision model")
                    }
                }
            } else {
                // Fallback: no vision inputs, this shouldn't happen for vision models with images
                candle_core::bail!("Vision model expected VisionModelInputs but got text inputs")
            }
        } else {
            // Text models: use standard ModelInputs
            let ModelInputs {
                input_ids,
                input_ids_full,
                seqlen_offsets,
                seqlen_offsets_full,
                context_lens,
                position_ids: _,
                paged_attn_meta,
                flash_meta,
                flash_meta_full,
            } = *inputs.downcast().expect("Downcast failed.");

            let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
                (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
                (Some(_), None) => {
                    candle_core::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
                }
                (None, Some(_)) => {
                    candle_core::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
                }
                (None, None) => None,
            };

            match self.model {
                Model::Llama(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
                Model::Phi2(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
                Model::XLoraLlama(ref model) => model.forward(
                    &input_ids,
                    input_ids_full.as_ref().unwrap_or(&input_ids),
                    &seqlen_offsets,
                    seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                    self.no_kv_cache,
                    &self.non_granular_state,
                    context_lens,
                    &flash_meta,
                    flash_meta_full.as_ref().unwrap_or(&flash_meta),
                )?,
                Model::Phi3(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, paged_attn_meta)?
                }
                Model::XLoraPhi3(ref model) => model.forward(
                    &input_ids,
                    input_ids_full.as_ref().unwrap_or(&input_ids),
                    &seqlen_offsets,
                    seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                    self.no_kv_cache,
                    &self.non_granular_state,
                    context_lens,
                    &flash_meta,
                    flash_meta_full.as_ref().unwrap_or(&flash_meta),
                )?,
                Model::Starcoder2(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, paged_attn_meta)?
                }
                Model::Qwen(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
                Model::Qwen3(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
                Model::Qwen3MoE(ref model) => {
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
                Model::Qwen3Vl(ref model) => {
                    // Text-only forward (no images in this request)
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
                Model::Qwen3VlMoE(ref model) => {
                    // Text-only forward (no images in this request)
                    model.forward(&input_ids, &seqlen_offsets, context_lens, paged_attn_meta)?
                }
            }
        };

        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        if self.model.is_vision() {
            ModelCategory::Vision {
                prefixer: Arc::new(GGUFVisionPrefixer),
            }
        } else {
            ModelCategory::Text
        }
    }
}

// TODO
impl AnyMoePipelineMixin for GGUFPipeline {}
