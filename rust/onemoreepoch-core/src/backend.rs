//! PyO3 bindings: `RustArray` (a Python-visible array) and `RustBackend`
//! (thin adapter over `array`/`ops`/`rng`, mirroring the Python `Backend` ABC).
//!
//! Deliberately excludes `im2col`/`col2im` — Conv2D's windowed-extraction
//! primitives aren't implemented natively here; the Python `RustBackend`
//! wrapper (`core/backend/rust_backend.py`) delegates those two to
//! `NumPyBackend` instead, converting through this array's flat/shape
//! representation. Everything else in the `Backend` contract is real Rust.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::array::{self, RustArray};
use crate::ops;
use crate::rng::Rng64;

fn to_py_err(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn extract_array(obj: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyRustArray>>() {
        Ok(pyref.inner.clone())
    } else if let Ok(value) = obj.extract::<f64>() {
        Ok(RustArray::scalar(value))
    } else {
        Err(PyTypeError::new_err("expected a RustArray or a number"))
    }
}

#[pyclass(name = "RustArray")]
#[derive(Clone)]
pub struct PyRustArray {
    pub inner: RustArray,
}

impl PyRustArray {
    fn wrap(inner: RustArray) -> Self {
        PyRustArray { inner }
    }
}

#[pymethods]
impl PyRustArray {
    #[staticmethod]
    fn from_flat(data: Vec<f64>, shape: Vec<usize>) -> Self {
        PyRustArray { inner: RustArray::new(data, shape) }
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        "float64"
    }

    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    fn tolist(&self) -> Vec<f64> {
        self.inner.data.clone()
    }

    fn item(&self) -> PyResult<f64> {
        if self.inner.data.len() != 1 {
            return Err(PyValueError::new_err("item() requires a single-element array"));
        }
        Ok(self.inner.data[0])
    }

    fn __float__(&self) -> PyResult<f64> {
        self.item()
    }

    fn __repr__(&self) -> String {
        format!("RustArray(shape={:?}, data={:?})", self.inner.shape, self.inner.data)
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x + y).map(Self::wrap).map_err(to_py_err)
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x - y).map(Self::wrap).map_err(to_py_err)
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&b, &self.inner, |x, y| x - y).map(Self::wrap).map_err(to_py_err)
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x * y).map(Self::wrap).map_err(to_py_err)
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x / y).map(Self::wrap).map_err(to_py_err)
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&b, &self.inner, |x, y| x / y).map(Self::wrap).map_err(to_py_err)
    }

    fn __neg__(&self) -> PyRustArray {
        Self::wrap(array::unary(&self.inner, |x| -x))
    }
}

#[pyclass]
pub struct RustBackend {
    rng: Rng64,
}

#[pymethods]
impl RustBackend {
    #[new]
    fn new() -> Self {
        RustBackend { rng: Rng64::new() }
    }

    // -- array creation ---------------------------------------------------

    fn zeros(&self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(shape, 0.0))
    }

    fn ones(&self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(shape, 1.0))
    }

    fn full(&self, shape: Vec<usize>, fill_value: f64) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(shape, fill_value))
    }

    fn zeros_like(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(a.inner.shape.clone(), 0.0))
    }

    fn ones_like(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(a.inner.shape.clone(), 1.0))
    }

    // -- random -------------------------------------------------------------

    fn seed(&mut self, value: u64) {
        self.rng.seed(value);
    }

    fn randn(&mut self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(self.rng.randn(&shape))
    }

    fn rand(&mut self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(self.rng.rand(&shape))
    }

    // -- arithmetic (each accepts RustArray or a plain number) -------------

    fn add(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x + y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn subtract(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x - y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn multiply(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x * y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn divide(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x / y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn negative(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| -x))
    }

    fn absolute(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.abs()))
    }

    fn power(&self, a: &Bound<'_, PyAny>, exponent: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, ee) = (extract_array(a)?, extract_array(exponent)?);
        array::broadcast_binary(&aa, &ee, |x, y| x.powf(y)).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn sqrt(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.sqrt()))
    }

    fn matmul(&self, a: &PyRustArray, b: &PyRustArray) -> PyResult<PyRustArray> {
        ops::matmul(&a.inner, &b.inner).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn exp(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.exp()))
    }

    fn log(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.ln()))
    }

    fn tanh(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.tanh()))
    }

    fn maximum(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, f64::max).map(PyRustArray::wrap).map_err(to_py_err)
    }

    fn greater(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| if x > y { 1.0 } else { 0.0 })
            .map(PyRustArray::wrap)
            .map_err(to_py_err)
    }

    // -- reductions ----------------------------------------------------------

    #[pyo3(signature = (a, axis=None, keepdims=false))]
    fn sum(&self, a: &PyRustArray, axis: Option<Vec<i64>>, keepdims: bool) -> PyRustArray {
        PyRustArray::wrap(ops::reduce_sum(&a.inner, axis.as_deref(), keepdims))
    }

    #[pyo3(signature = (a, axis=None, keepdims=false))]
    fn mean(&self, a: &PyRustArray, axis: Option<Vec<i64>>, keepdims: bool) -> PyRustArray {
        PyRustArray::wrap(ops::reduce_mean(&a.inner, axis.as_deref(), keepdims))
    }

    #[pyo3(signature = (a, axis=None, keepdims=false))]
    fn max(&self, a: &PyRustArray, axis: Option<Vec<i64>>, keepdims: bool) -> PyRustArray {
        PyRustArray::wrap(ops::reduce_max(&a.inner, axis.as_deref(), keepdims))
    }

    // -- shape -----------------------------------------------------------

    fn reshape(&self, a: &PyRustArray, shape: Vec<usize>) -> PyResult<PyRustArray> {
        let size: usize = shape.iter().product();
        if size != a.inner.size() {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {} into shape {:?}",
                a.inner.size(),
                shape
            )));
        }
        Ok(PyRustArray::wrap(RustArray::new(a.inner.data.clone(), shape)))
    }

    #[pyo3(signature = (a, axes=None))]
    fn transpose(&self, a: &PyRustArray, axes: Option<Vec<usize>>) -> PyRustArray {
        PyRustArray::wrap(ops::transpose(&a.inner, axes.as_deref()))
    }

    fn broadcast_to(&self, a: &PyRustArray, shape: Vec<usize>) -> PyResult<PyRustArray> {
        array::broadcast_to(&a.inner, &shape).map(PyRustArray::wrap).map_err(to_py_err)
    }
}
