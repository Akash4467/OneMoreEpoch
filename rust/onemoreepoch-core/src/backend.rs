use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::array::{self, RustArray};
use crate::ops;
use crate::rng::Rng64;

// Wraps a string error into a Python ValueError
fn to_py_err(message: String) -> PyErr {
    PyValueError::new_err(message)
}

// Extracts a RustArray from a Python RustArray or plain number
fn extract_array(obj: &Bound<'_, PyAny>) -> PyResult<RustArray> {
    if let Ok(pyref) = obj.extract::<PyRef<'_, PyRustArray>>() {
        Ok(pyref.inner.clone())
    } else if let Ok(value) = obj.extract::<f64>() {
        Ok(RustArray::scalar(value))
    } else {
        Err(PyTypeError::new_err("expected a RustArray or a number"))
    }
}

// Python-visible wrapper around a RustArray
#[pyclass(name = "RustArray")]
#[derive(Clone)]
pub struct PyRustArray {
    pub inner: RustArray,
}

// Internal constructor for PyRustArray
impl PyRustArray {
    // Wraps a RustArray as a PyRustArray
    fn wrap(inner: RustArray) -> Self {
        PyRustArray { inner }
    }
}

// Python-exposed methods and operators for RustArray
#[pymethods]
impl PyRustArray {
    // Builds a RustArray from a flat buffer and shape
    #[staticmethod]
    fn from_flat(data: Vec<f64>, shape: Vec<usize>) -> Self {
        PyRustArray { inner: RustArray::new(data, shape) }
    }

    // Returns the array's shape
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    // Returns the array's number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    // Returns the array's dtype label
    #[getter]
    fn dtype(&self) -> &'static str {
        "float64"
    }

    // Returns the array's total element count
    #[getter]
    fn size(&self) -> usize {
        self.inner.size()
    }

    // Returns the array's data as a flat Python list
    fn tolist(&self) -> Vec<f64> {
        self.inner.data.clone()
    }

    // Returns the single scalar value of a one-element array
    fn item(&self) -> PyResult<f64> {
        if self.inner.data.len() != 1 {
            return Err(PyValueError::new_err("item() requires a single-element array"));
        }
        Ok(self.inner.data[0])
    }

    // Converts a one-element array to a Python float
    fn __float__(&self) -> PyResult<f64> {
        self.item()
    }

    // Returns a debug string representation of the array
    fn __repr__(&self) -> String {
        format!("RustArray(shape={:?}, data={:?})", self.inner.shape, self.inner.data)
    }

    // Adds another array or number, elementwise with broadcasting
    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x + y).map(Self::wrap).map_err(to_py_err)
    }

    // Reflected addition
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        self.__add__(other)
    }

    // Subtracts another array or number, elementwise with broadcasting
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x - y).map(Self::wrap).map_err(to_py_err)
    }

    // Reflected subtraction
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&b, &self.inner, |x, y| x - y).map(Self::wrap).map_err(to_py_err)
    }

    // Multiplies by another array or number, elementwise with broadcasting
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x * y).map(Self::wrap).map_err(to_py_err)
    }

    // Reflected multiplication
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        self.__mul__(other)
    }

    // Divides by another array or number, elementwise with broadcasting
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&self.inner, &b, |x, y| x / y).map(Self::wrap).map_err(to_py_err)
    }

    // Reflected division
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let b = extract_array(other)?;
        array::broadcast_binary(&b, &self.inner, |x, y| x / y).map(Self::wrap).map_err(to_py_err)
    }

    // Negates the array elementwise
    fn __neg__(&self) -> PyRustArray {
        Self::wrap(array::unary(&self.inner, |x| -x))
    }
}

// Python-exposed computational backend adapting array/ops/rng to the Backend ABC
#[pyclass]
pub struct RustBackend {
    rng: Rng64,
}

// Python-exposed methods for RustBackend
#[pymethods]
impl RustBackend {
    // Builds a RustBackend with a fresh RNG
    #[new]
    fn new() -> Self {
        RustBackend { rng: Rng64::new() }
    }

    // Builds an array of the given shape filled with zeros
    fn zeros(&self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(shape, 0.0))
    }

    // Builds an array of the given shape filled with ones
    fn ones(&self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(shape, 1.0))
    }

    // Builds an array of the given shape filled with a constant
    fn full(&self, shape: Vec<usize>, fill_value: f64) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(shape, fill_value))
    }

    // Builds a zero-filled array matching another array's shape
    fn zeros_like(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(a.inner.shape.clone(), 0.0))
    }

    // Builds a one-filled array matching another array's shape
    fn ones_like(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(RustArray::filled(a.inner.shape.clone(), 1.0))
    }

    // Reseeds the backend's RNG for reproducible sampling
    fn seed(&mut self, value: u64) {
        self.rng.seed(value);
    }

    // Samples a standard-normal array of the given shape
    fn randn(&mut self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(self.rng.randn(&shape))
    }

    // Samples a uniform [0, 1) array of the given shape
    fn rand(&mut self, shape: Vec<usize>) -> PyRustArray {
        PyRustArray::wrap(self.rng.rand(&shape))
    }

    // Adds two arrays or numbers elementwise with broadcasting
    fn add(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x + y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Subtracts two arrays or numbers elementwise with broadcasting
    fn subtract(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x - y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Multiplies two arrays or numbers elementwise with broadcasting
    fn multiply(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x * y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Divides two arrays or numbers elementwise with broadcasting
    fn divide(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| x / y).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Negates an array elementwise
    fn negative(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| -x))
    }

    // Takes the elementwise absolute value of an array
    fn absolute(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.abs()))
    }

    // Raises an array to a power, elementwise with broadcasting
    fn power(&self, a: &Bound<'_, PyAny>, exponent: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, ee) = (extract_array(a)?, extract_array(exponent)?);
        array::broadcast_binary(&aa, &ee, |x, y| x.powf(y)).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Takes the elementwise square root of an array
    fn sqrt(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.sqrt()))
    }

    // Computes matrix multiplication between two arrays
    fn matmul(&self, a: &PyRustArray, b: &PyRustArray) -> PyResult<PyRustArray> {
        ops::matmul(&a.inner, &b.inner).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Takes the elementwise exponential of an array
    fn exp(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.exp()))
    }

    // Takes the elementwise natural log of an array
    fn log(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.ln()))
    }

    // Takes the elementwise hyperbolic tangent of an array
    fn tanh(&self, a: &PyRustArray) -> PyRustArray {
        PyRustArray::wrap(array::unary(&a.inner, |x| x.tanh()))
    }

    // Takes the elementwise maximum of two arrays or numbers with broadcasting
    fn maximum(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, f64::max).map(PyRustArray::wrap).map_err(to_py_err)
    }

    // Computes an elementwise greater-than comparison with broadcasting
    fn greater(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyRustArray> {
        let (aa, bb) = (extract_array(a)?, extract_array(b)?);
        array::broadcast_binary(&aa, &bb, |x, y| if x > y { 1.0 } else { 0.0 })
            .map(PyRustArray::wrap)
            .map_err(to_py_err)
    }

    // Reduces an array by summation over the given axes
    #[pyo3(signature = (a, axis=None, keepdims=false))]
    fn sum(&self, a: &PyRustArray, axis: Option<Vec<i64>>, keepdims: bool) -> PyRustArray {
        PyRustArray::wrap(ops::reduce_sum(&a.inner, axis.as_deref(), keepdims))
    }

    // Reduces an array by averaging over the given axes
    #[pyo3(signature = (a, axis=None, keepdims=false))]
    fn mean(&self, a: &PyRustArray, axis: Option<Vec<i64>>, keepdims: bool) -> PyRustArray {
        PyRustArray::wrap(ops::reduce_mean(&a.inner, axis.as_deref(), keepdims))
    }

    // Reduces an array by taking the maximum over the given axes
    #[pyo3(signature = (a, axis=None, keepdims=false))]
    fn max(&self, a: &PyRustArray, axis: Option<Vec<i64>>, keepdims: bool) -> PyRustArray {
        PyRustArray::wrap(ops::reduce_max(&a.inner, axis.as_deref(), keepdims))
    }

    // Reshapes an array to a new shape with the same element count
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

    // Permutes an array's axes, defaulting to a full reversal
    #[pyo3(signature = (a, axes=None))]
    fn transpose(&self, a: &PyRustArray, axes: Option<Vec<usize>>) -> PyRustArray {
        PyRustArray::wrap(ops::transpose(&a.inner, axes.as_deref()))
    }

    // Materializes an array broadcast to a larger target shape
    fn broadcast_to(&self, a: &PyRustArray, shape: Vec<usize>) -> PyResult<PyRustArray> {
        array::broadcast_to(&a.inner, &shape).map(PyRustArray::wrap).map_err(to_py_err)
    }
}
