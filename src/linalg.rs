use ndarray::{Array1, ArrayView2};
use ndarray_linalg::{Eigh, UPLO};

pub fn eigen(m: &ArrayView2<f64>) -> Array1<f64> {
    assert_eq!(m.shape()[0], m.shape()[1], "The matrix must be square");

    let (eigen, _) = m.eigh(UPLO::Upper).unwrap();

    eigen
}
