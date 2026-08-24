//! `onemoreepoch_core`: the Rust computational backend, exposed to Python
//! via PyO3 as `onemoreepoch._rustcore`. Scope for this pass is the same
//! low-level array-primitive contract `NumPyBackend` satisfies — full
//! autograd graph traversal in Rust (doc §7.2's eventual end state) is a
//! distinct, later phase and stays in Python for now.
//!
//! The doc's fuller future `src/` layout (`tensor/`, `autograd/`, `memory/`,
//! `device/`, `errors/`) is deliberately not scaffolded as empty
//! directories here — they'll appear when there's real code to put in
//! them, not before:
//!   tensor/    — a Rust-owned Tensor type
//!   autograd/  — full graph traversal in Rust
//!   memory/    — custom allocation / pooling
//!   device/    — GPU device abstraction
//!   errors/    — dedicated Rust exception types

mod array;
mod backend;
mod ops;
mod rng;

use pyo3::prelude::*;

#[pymodule]
fn _rustcore(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<backend::PyRustArray>()?;
    m.add_class::<backend::RustBackend>()?;
    Ok(())
}
