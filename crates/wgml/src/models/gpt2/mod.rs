//! GPT-2 inference on the GPU or CPU.

pub use tokenizer::Tokenizer;

pub mod cpu;
mod tokenizer;
pub mod transformer;
