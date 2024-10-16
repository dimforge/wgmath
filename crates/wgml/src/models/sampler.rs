//! A CPU-side sampler.

use crate::ops::SoftMax;
use nalgebra::{Dyn, StorageMut, Vector};
use ordered_float::OrderedFloat;
use rand::random;

#[derive(Default, Copy, Clone)]
pub struct ProbIndex {
    prob: f32,
    index: usize,
}

pub struct Sampler {
    vocab_size: usize,
    prob_index: Vec<ProbIndex>, // buffer used in top-p sampling.
    temperature: f32,
    topp: f32,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32) -> Self {
        Self {
            vocab_size,
            prob_index: vec![ProbIndex::default(); vocab_size],
            temperature,
            topp,
        }
    }

    // Sample the token given the logits and some hyperparameters.
    pub fn sample<S: StorageMut<f32, Dyn>>(
        &mut self,
        mut logits: &mut Vector<f32, Dyn, S>,
    ) -> usize {
        if self.temperature == 0.0 {
            // Greedy argmax sampling: take the token with the highest probability.
            Self::sample_argmax(logits)
        } else {
            // Apply the temperature to the logits.
            *logits /= self.temperature;

            // Apply softmax to the logits to get probabilities for next token.
            SoftMax::run_cpu(&mut logits);
            let coin = random();

            if self.topp <= 0.0 || self.topp >= 1.0 {
                // Simply sample from the predicted probability distribution.
                Self::sample_mult(logits, coin)
            } else {
                // Top-p (nucleus) sampling, clamping the least likely tokens to zero.
                self.sample_topp(logits, coin)
            }
        }
    }

    pub fn sample_argmax<S: StorageMut<f32, Dyn>>(probabilities: &Vector<f32, Dyn, S>) -> usize {
        probabilities.imax()
    }

    /// Sample index from probabilities.
    ///
    /// The `probabilities` must sum to 1.0.
    /// `coin` is a random number in [0, 1).
    pub fn sample_mult<S: StorageMut<f32, Dyn>>(
        probabilities: &Vector<f32, Dyn, S>,
        coin: f32,
    ) -> usize {
        let mut cdf = 0.0;

        for (i, prob) in probabilities.iter().enumerate() {
            cdf += *prob;
            if coin < cdf {
                return i;
            }
        }

        return probabilities.len() - 1;
    }

    /// Top-p sampling (or "nucleus sampling") samples from the smallest set of tokens
    /// that exceed probability topp.
    ///
    /// This way we never sample token that have very low probabilities and are less likely
    /// to go "off the rails". Coin is a random number in [0, 1).
    pub fn sample_topp<S: StorageMut<f32, Dyn>>(
        &mut self,
        probabilities: &Vector<f32, Dyn, S>,
        coin: f32,
    ) -> usize {
        let mut n0 = 0;

        // Quicksort indices in descending order of probabilities.
        // Values smaller than (1 - topp) / (n - 1) cannot be part of the result.
        // So for efficiency we crop these out as candidates before sorting.
        let cutoff = (1.0 - self.topp) / (self.vocab_size as f32 - 1.0);

        for i in 0..probabilities.len() {
            if probabilities[i] >= cutoff {
                self.prob_index[n0].index = i;
                self.prob_index[n0].prob = probabilities[i];
                n0 += 1;
            }
        }

        // Sort in decreasing order.
        self.prob_index[..n0].sort_by_key(|pid| OrderedFloat(-pid.prob));

        // Truncate the list where cumulative probability exceeds the list.
        let mut cumulative_prob = 0.0;
        let mut last_idx = n0 - 1; // In case of rounding errors consider all elements.

        for i in 0..n0 {
            cumulative_prob += self.prob_index[i].prob;
            if cumulative_prob > self.topp {
                last_idx = i;
                break;
            }
        }

        // Sample from the truncated list.
        let r = coin * cumulative_prob;
        let mut cdf = 0.0;

        for i in 0..=last_idx {
            cdf += self.prob_index[i].prob;
            if r < cdf {
                return self.prob_index[i].index;
            }
        }

        return self.prob_index[last_idx].index; // In case of rounding errors.
    }
}
