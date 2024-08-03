use std::ops::AddAssign;

use ndarray::{Array1, ArrayView1, ErrorKind, ShapeError};
use num_traits::Float;

use crate::{check_lengths, check_total, utils::one_nan_to_num, Winner};

pub fn elo<A: Float + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    total: usize,
    initial: A,
    base: A,
    scale: A,
    k: A,
) -> Result<Array1<A>, ShapeError> {
    check_lengths!(xs.len(), ys.len(), ws.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    check_total!(xs, ys, total);

    let mut scores = Array1::from_elem(total, initial);

    let half = A::one() / (A::one() + A::one());

    for ((x, y), &ref w) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        let q_x = one_nan_to_num(base.powf(scores[*x] / scale), A::zero());
        let q_y = one_nan_to_num(base.powf(scores[*y] / scale), A::zero());

        let q = one_nan_to_num(q_x + q_y, A::zero());

        let expected_x = one_nan_to_num(q_x / q, A::zero());
        let expected_y = one_nan_to_num(q_y / q, A::zero());

        let (scored_x, scored_y) = match w {
            Winner::X => (A::one(), A::zero()),
            Winner::Y => (A::zero(), A::one()),
            Winner::Draw => (half, half),
            _ => continue,
        };

        scores[*x] += k * (scored_x - expected_x);
        scores[*y] += k * (scored_y - expected_y);
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_elo() {
        let xs = array![3, 2, 1, 0];
        let ys = array![0, 1, 2, 3];
        let ws = array![Winner::X, Winner::Draw, Winner::Y, Winner::X];
        let initial: f64 = 1500.0;
        let base: f64 = 10.0;
        let scale: f64 = 400.0;
        let k: f64 = 30.0;

        let expected = array![1501.0, 1485.0, 1515.0, 1498.0];

        let actual = elo(
            &xs.view(),
            &ys.view(),
            &ws.view(),
            5,
            initial,
            base,
            scale,
            k,
        )
        .unwrap();

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-0, "a = {}, b = {}", a, b);
        }
    }
}
