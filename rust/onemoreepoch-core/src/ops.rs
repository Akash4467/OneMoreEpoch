//! Pure-Rust algorithms over RustArray — no PyO3, cargo-testable directly.

use crate::array::{strides_for, unravel, RustArray};

pub fn matmul(a: &RustArray, b: &RustArray) -> Result<RustArray, String> {
    match (a.shape.len(), b.shape.len()) {
        (2, 2) => {
            let (m, k1) = (a.shape[0], a.shape[1]);
            let (k2, n) = (b.shape[0], b.shape[1]);
            if k1 != k2 {
                return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape, b.shape));
            }
            let mut data = vec![0.0f64; m * n];
            for i in 0..m {
                for k in 0..k1 {
                    let av = a.data[i * k1 + k];
                    if av == 0.0 {
                        continue;
                    }
                    for j in 0..n {
                        data[i * n + j] += av * b.data[k * n + j];
                    }
                }
            }
            Ok(RustArray { data, shape: vec![m, n] })
        }
        (2, 1) => {
            let (m, k1) = (a.shape[0], a.shape[1]);
            let k2 = b.shape[0];
            if k1 != k2 {
                return Err(format!("matmul shape mismatch: {:?} @ {:?}", a.shape, b.shape));
            }
            let mut data = vec![0.0f64; m];
            for i in 0..m {
                let mut acc = 0.0;
                for k in 0..k1 {
                    acc += a.data[i * k1 + k] * b.data[k];
                }
                data[i] = acc;
            }
            Ok(RustArray { data, shape: vec![m] })
        }
        _ => Err(format!(
            "matmul only supports 2D@2D or 2D@1D, got shapes {:?} and {:?}",
            a.shape, b.shape
        )),
    }
}

fn normalize_axes(axes: Option<&[i64]>, ndim: usize) -> Vec<usize> {
    match axes {
        None => (0..ndim).collect(),
        Some(list) => list
            .iter()
            .map(|&ax| if ax < 0 { (ax + ndim as i64) as usize } else { ax as usize })
            .collect(),
    }
}

fn reduce_generic(
    a: &RustArray,
    axes: Option<&[i64]>,
    keepdims: bool,
    init: f64,
    combine: impl Fn(f64, f64) -> f64,
    finalize: impl Fn(f64, usize) -> f64,
) -> RustArray {
    let ndim = a.shape.len();
    let reduce_axes = normalize_axes(axes, ndim);

    let out_shape_full: Vec<usize> = (0..ndim)
        .map(|i| if reduce_axes.contains(&i) { 1 } else { a.shape[i] })
        .collect();
    let out_size: usize = out_shape_full.iter().product::<usize>().max(1);
    let out_strides = strides_for(&out_shape_full);

    let mut acc = vec![init; out_size];
    let mut counts = vec![0usize; out_size];

    for flat in 0..a.data.len() {
        let idx = unravel(flat, &a.shape);
        let mut out_idx = idx.clone();
        for &ax in &reduce_axes {
            out_idx[ax] = 0;
        }
        let out_flat: usize = (0..ndim).map(|i| out_idx[i] * out_strides[i]).sum();
        acc[out_flat] = combine(acc[out_flat], a.data[flat]);
        counts[out_flat] += 1;
    }

    let data: Vec<f64> = acc.iter().zip(counts.iter()).map(|(&v, &c)| finalize(v, c)).collect();
    let final_shape = if keepdims {
        out_shape_full
    } else {
        out_shape_full
            .iter()
            .enumerate()
            .filter(|(i, _)| !reduce_axes.contains(i))
            .map(|(_, &d)| d)
            .collect()
    };
    RustArray { data, shape: final_shape }
}

pub fn reduce_sum(a: &RustArray, axes: Option<&[i64]>, keepdims: bool) -> RustArray {
    reduce_generic(a, axes, keepdims, 0.0, |acc, x| acc + x, |v, _| v)
}

pub fn reduce_mean(a: &RustArray, axes: Option<&[i64]>, keepdims: bool) -> RustArray {
    reduce_generic(a, axes, keepdims, 0.0, |acc, x| acc + x, |v, c| v / c.max(1) as f64)
}

pub fn reduce_max(a: &RustArray, axes: Option<&[i64]>, keepdims: bool) -> RustArray {
    reduce_generic(a, axes, keepdims, f64::NEG_INFINITY, f64::max, |v, _| v)
}

pub fn transpose(a: &RustArray, axes: Option<&[usize]>) -> RustArray {
    let ndim = a.shape.len();
    let perm: Vec<usize> = match axes {
        Some(p) => p.to_vec(),
        None => (0..ndim).rev().collect(),
    };
    let new_shape: Vec<usize> = perm.iter().map(|&i| a.shape[i]).collect();
    let new_strides = strides_for(&new_shape);
    let mut data = vec![0.0f64; a.data.len()];
    for flat in 0..a.data.len() {
        let idx = unravel(flat, &a.shape);
        let new_flat: usize = (0..ndim).map(|j| idx[perm[j]] * new_strides[j]).sum();
        data[new_flat] = a.data[flat];
    }
    RustArray { data, shape: new_shape }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>, shape: Vec<usize>) -> RustArray {
        RustArray { data, shape }
    }

    #[test]
    fn matmul_2d_2d() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = arr(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let out = matmul(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(out.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_2d_1d() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = arr(vec![1.0, 1.0], vec![2]);
        let out = matmul(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.data, vec![3.0, 7.0]);
    }

    #[test]
    fn matmul_dimension_mismatch_errors() {
        let a = arr(vec![1.0, 2.0], vec![1, 2]);
        let b = arr(vec![1.0, 2.0, 3.0], vec![3, 1]);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn sum_all_axes() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = reduce_sum(&a, None, false);
        assert_eq!(out.shape, Vec::<usize>::new());
        assert_eq!(out.data, vec![10.0]);
    }

    #[test]
    fn sum_axis_keepdims() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&a, Some(&[1]), true);
        assert_eq!(out.shape, vec![2, 1]);
        assert_eq!(out.data, vec![6.0, 15.0]);
    }

    #[test]
    fn sum_axis_no_keepdims_drops_axis() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = reduce_sum(&a, Some(&[0]), false);
        assert_eq!(out.shape, vec![3]);
        assert_eq!(out.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn mean_matches_manual() {
        let a = arr(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
        let out = reduce_mean(&a, None, false);
        assert_eq!(out.data, vec![5.0]);
    }

    #[test]
    fn max_axis() {
        let a = arr(vec![1.0, 5.0, 3.0, 2.0, 0.0, 9.0], vec![2, 3]);
        let out = reduce_max(&a, Some(&[1]), false);
        assert_eq!(out.shape, vec![2]);
        assert_eq!(out.data, vec![5.0, 9.0]);
    }

    #[test]
    fn transpose_2d_is_matrix_transpose() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = transpose(&a, None);
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(out.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_with_explicit_axes() {
        let a = arr((0..24).map(|x| x as f64).collect(), vec![2, 3, 4]);
        let out = transpose(&a, Some(&[1, 0, 2]));
        assert_eq!(out.shape, vec![3, 2, 4]);
        // spot check: out[1][0][2] should equal a[0][1][2] = 6.0
        let idx = 1 * (2 * 4) + 0 * 4 + 2;
        assert_eq!(out.data[idx], 6.0);
    }

    #[test]
    fn transpose_roundtrip() {
        let a = arr((0..12).map(|x| x as f64).collect(), vec![3, 4]);
        let t = transpose(&a, None);
        let back = transpose(&t, None);
        assert_eq!(back, a);
    }
}
