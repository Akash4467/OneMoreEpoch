mod array;
mod backend;
mod ops;
mod rng;

use pyo3::prelude::*;

// Registers the RustArray and RustBackend classes with the Python module
#[pymodule]
fn _rustcore(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<backend::PyRustArray>()?;
    m.add_class::<backend::RustBackend>()?;
    Ok(())
}
