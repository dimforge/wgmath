// NOTE: similar to https://raw.githubusercontent.com/rahoua/pecca-rs/main/src/llama2/tokenizer.rs
//       but adjusted to load from gguf.

use crate::gguf::Gguf;
use std::collections::HashMap;
use std::fmt;

pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    vocab_index: HashMap<String, usize>,
    byte_pieces: Vec<char>, // stores all single-byte strings
}

impl Tokenizer {
    /// Token id for the unknown token.
    const UNKNOWN: usize = 0;
    /// Token id for the beginning of sequence.
    const BOS: usize = 1;
    /// Token id for the end of sequence.
    const EOS: usize = 2;

    pub fn from_gguf(gguf: &Gguf) -> Tokenizer {
        let vocab_scores = gguf.metadata["tokenizer.ggml.scores"]
            .as_f32_array()
            .to_vec();
        let vocab = gguf.metadata["tokenizer.ggml.tokens"]
            .as_string_array()
            .to_vec();
        let byte_pieces: Vec<char> = (0..=256).map(|i| i as u8 as char).collect();

        let mut vocab_index = HashMap::new();
        for n in 0..vocab.len() {
            vocab_index.insert(vocab[n].clone(), n);
        }

        Tokenizer {
            vocab,
            vocab_scores,
            vocab_index,
            byte_pieces,
        }
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<usize> {
        let mut tokens: Vec<usize> = Vec::new();
        if bos {
            tokens.push(Self::BOS);
        }

        // Comment from llama.c:
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        //       energy to read more of the sentencepiece code to figure out what it's doing
        if !text.is_empty() {
            let dummy_prefix = self.vocab_index.get(" ").copied().unwrap_or(Self::UNKNOWN);
            tokens.push(dummy_prefix);
        }

        for ch in text.chars() {
            let ch_str = ch.to_string();
            match self.vocab_index.get(&ch_str) {
                Some(&id) => tokens.push(id),
                None => {
                    // byte_fallback encoding: just encode each byte as a token
                    // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                    // so the individual bytes only start at index 3
                    for byte in ch_str.as_bytes() {
                        tokens.push(*byte as usize + 3);
                    }
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = 0;
            let mut best_idx = None;

            for i in 0..(tokens.len() - 1) {
                let pair = format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
                if let Some(&id) = self.vocab_index.get(&pair) {
                    if self.vocab_scores[id] > best_score {
                        best_score = self.vocab_scores[id];
                        best_id = id;
                        best_idx = Some(i);
                    }
                }
            }

            if let Some(idx) = best_idx {
                tokens[idx] = best_id;
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(Self::EOS);
        }

        tokens
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        let mut piece = self.vocab[token].as_str();
        if prev_token == 1 {
            piece = piece.trim_start();
        }
        if let Some(hex) = piece.strip_prefix("<0x") {
            if let Ok(byte) = usize::from_str_radix(&hex[..2], 16) {
                return self.byte_pieces[byte].to_string();
            }
        }
        piece.replace("▁", " ").to_string()
    }
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tokenizer with vocab size: {}", self.vocab.len())
    }
}
