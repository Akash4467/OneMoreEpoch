// Flat-buffer row-major array with shape
#[derive(Clone, Debug, PartialEq)]
pub struct RustArray {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

// Constructors and basic queries for RustArray
impl RustArray {
    // Builds a RustArray from an existing flat buffer and shape
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        RustArray { data, shape }
    }

    // Builds a zero-dimensional single-value RustArray
    pub fn scalar(value: f64) -> Self {
        RustArray { data: vec![value], shape: vec![] }
    }

    // Builds a RustArray of the given shape filled with one value
    pub fn filled(shape: Vec<usize>, value: f64) -> Self {
        let size: usize = shape.iter().product();
        RustArray { data: vec![value; size], shape }
    }

    // Returns the total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    // Returns the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

// Computes row-major strides for a shape
pub fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Converts a flat index into per-axis indices for a shape
pub fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let strides = strides_for(shape);
    let mut idx = vec![0usize; shape.len()];
    for i in 0..shape.len() {
        idx[i] = flat / strides[i].max(1);
        flat %= strides[i].max(1);
    }
    idx
}

// Computes the numpy-style broadcast result shape for two shapes
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let d = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(format!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                a, b
            ));
        };
        out[rank - 1 - i] = d;
    }
    Ok(out)
}

// Maps an output index back to a flat index in a broadcast input's buffer
fn broadcast_index(out_idx: &[usize], in_shape: &[usize]) -> usize {
    let rank = out_idx.len();
    let offset = rank - in_shape.len();
    let strides = strides_for(in_shape);
    let mut flat = 0usize;
    for i in 0..in_shape.len() {
        let component = if in_shape[i] == 1 { 0 } else { out_idx[offset + i] };
        flat += component * strides[i];
    }
    flat
}

// Applies a binary elementwise function over two arrays with broadcasting
pub fn broadcast_binary<F: Fn(f64, f64) -> f64>(
    a: &RustArray,
    b: &RustArray,
    f: F,
) -> Result<RustArray, String> {
    let out_shape = broadcast_shapes(&a.shape, &b.shape)?;
    let size: usize = out_shape.iter().product();
    let mut data = Vec::with_capacity(size);
    for flat in 0..size {
        let idx = unravel(flat, &out_shape);
        let ai = broadcast_index(&idx, &a.shape);
        let bi = broadcast_index(&idx, &b.shape);
        data.push(f(a.data[ai], b.data[bi]));
    }
    Ok(RustArray { data, shape: out_shape })
}

// Applies a unary elementwise function over an array
pub fn unary(a: &RustArray, f: impl Fn(f64) -> f64) -> RustArray {
    RustArray {
        data: a.data.iter().map(|&x| f(x)).collect(),
        shape: a.shape.clone(),
    }
}

// Materializes an array broadcast to a larger target shape
pub fn broadcast_to(a: &RustArray, target_shape: &[usize]) -> Result<RustArray, String> {
    let rank = target_shape.len();
    if a.shape.len() > rank {
        return Err(format!("cannot broadcast shape {:?} to {:?}", a.shape, target_shape));
    }
    let offset = rank - a.shape.len();
    for i in 0..a.shape.len() {
        if a.shape[i] != 1 && a.shape[i] != target_shape[offset + i] {
            return Err(format!("cannot broadcast shape {:?} to {:?}", a.shape, target_shape));
        }
    }
    let size: usize = target_shape.iter().product();
    let mut data = Vec::with_capacity(size);
    for flat in 0..size {
        let idx = unravel(flat, target_shape);
        let ai = broadcast_index(&idx, &a.shape);
        data.push(a.data[ai]);
    }
    Ok(RustArray { data, shape: target_shape.to_vec() })
}

// Unit tests for array primitives
#[cfg(test)]
mod tests {
    use super::*;

    // Checks broadcast_shapes follows numpy's broadcasting rules
    #[test]
    fn broadcast_shapes_matches_numpy_rules() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 1]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[], &[3, 4]).unwrap(), vec![3, 4]);
        assert!(broadcast_shapes(&[3, 4], &[3, 5]).is_err());
    }

    // Checks elementwise add with trailing-dimension broadcasting
    #[test]
    fn broadcast_binary_add_with_trailing_broadcast() {
        let a = RustArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = RustArray::new(vec![10.0, 20.0, 30.0], vec![3]);
        let out = broadcast_binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    // Checks elementwise multiply against a scalar array
    #[test]
    fn broadcast_binary_with_scalar() {
        let a = RustArray::new(vec![1.0, 2.0, 3.0], vec![3]);
        let scalar = RustArray::scalar(10.0);
        let out = broadcast_binary(&a, &scalar, |x, y| x * y).unwrap();
        assert_eq!(out.data, vec![10.0, 20.0, 30.0]);
    }

    // Checks broadcast_to expands a leading axis correctly
    #[test]
    fn broadcast_to_expands_leading_axis() {
        let a = RustArray::new(vec![1.0, 2.0, 3.0], vec![3]);
        let out = broadcast_to(&a, &[2, 3]).unwrap();
        assert_eq!(out.data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    // Checks broadcast_to rejects an incompatible target shape
    #[test]
    fn broadcast_to_rejects_incompatible_shape() {
        let a = RustArray::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(broadcast_to(&a, &[2, 4]).is_err());
    }

    // Checks unravel and strides_for are consistent with each other
    #[test]
    fn unravel_and_strides_roundtrip() {
        let shape = vec![2, 3, 4];
        let strides = strides_for(&shape);
        assert_eq!(strides, vec![12, 4, 1]);
        let idx = unravel(17, &shape);
        assert_eq!(idx, vec![1, 1, 1]);
    }
}
