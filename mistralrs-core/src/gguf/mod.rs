mod chat_template;
mod content;
mod gguf_tokenizer;
use strum::EnumString;

use anyhow::{Context, Result};
pub(crate) use chat_template::get_gguf_chat_template;
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{convert_gguf_to_hf_tokenizer, GgufTokenizerConversion};
use std::str::FromStr;

pub const GGUF_MULTI_FILE_DELIMITER: &str = " ";

#[derive(Debug, EnumString, Clone, Copy, strum::Display)]
#[strum(serialize_all = "lowercase")]
pub enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
    Starcoder2,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    // Vision-capable architectures
    #[strum(serialize = "qwen3vl")]
    Qwen3Vl,
    #[strum(serialize = "qwen3vlmoe")]
    Qwen3VlMoE,
    // Vision encoder architecture (for mmproj files)
    #[strum(serialize = "clip")]
    Clip,
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
impl GGUFArchitecture {
    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        Self::from_str(&value.as_ref().to_ascii_lowercase())
            .with_context(|| format!("Unknown GGUF architecture `{value}`"))
            .map_err(anyhow::Error::msg)
    }

    /// Returns true if this architecture supports vision/multimodal inputs.
    /// Vision architectures require an mmproj file for the vision encoder.
    pub fn supports_vision(&self) -> bool {
        matches!(self, Self::Qwen3Vl | Self::Qwen3VlMoE)
    }

    /// Returns true if this is a vision encoder architecture (for mmproj files).
    pub fn is_vision_encoder(&self) -> bool {
        matches!(self, Self::Clip)
    }
}
