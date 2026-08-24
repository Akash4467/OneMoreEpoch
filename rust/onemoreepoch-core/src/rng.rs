//! Seedable RNG for `rand`/`randn`, matching NumPyBackend's contract
//! (unseeded = OS entropy; `seed()` makes it reproducible).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::array::RustArray;

pub struct Rng64 {
    rng: StdRng,
}

impl Rng64 {
    pub fn new() -> Self {
        Rng64 { rng: StdRng::from_entropy() }
    }

    pub fn seed(&mut self, value: u64) {
        self.rng = StdRng::seed_from_u64(value);
    }

    pub fn rand(&mut self, shape: &[usize]) -> RustArray {
        let size: usize = shape.iter().product();
        let data: Vec<f64> = (0..size).map(|_| self.rng.gen::<f64>()).collect();
        RustArray::new(data, shape.to_vec())
    }

    /// Box-Muller transform — two independent standard-normal samples per pair.
    pub fn randn(&mut self, shape: &[usize]) -> RustArray {
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        while data.len() < size {
            let u1: f64 = self.rng.gen::<f64>().max(1e-12);
            let u2: f64 = self.rng.gen::<f64>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data.push(r * theta.cos());
            if data.len() < size {
                data.push(r * theta.sin());
            }
        }
        RustArray::new(data, shape.to_vec())
    }
}
