use std::ops::AddAssign;

use ndarray::{Array1, ArrayView1, ErrorKind, ShapeError};
use num_traits::Float;

use crate::utils::nan_to_num;
use crate::{check_lengths, check_total, utils::one_nan_to_num, Winner};

pub fn elo<A: Float + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    winners: &ArrayView1<Winner>,
    weights: &ArrayView1<A>,
    total: usize,
    initial: A,
    base: A,
    scale: A,
    k: A,
    win_weight: A,
    tie_weight: A,
) -> Result<Array1<A>, ShapeError> {
    check_lengths!(xs.len(), ys.len(), winners.len(), weights.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    check_total!(total, xs, ys);

    let mut scores = Array1::from_elem(total, initial);

    for (((&x, &y), &ref w), &weight) in xs.iter().zip(ys.iter()).zip(winners.iter()).zip(weights) {
        let q_x = one_nan_to_num(base.powf(scores[x] / scale), A::zero());
        let q_y = one_nan_to_num(base.powf(scores[y] / scale), A::zero());

        let q = one_nan_to_num(q_x + q_y, A::zero());

        let expected_x = one_nan_to_num(q_x / q, A::zero());
        let expected_y = one_nan_to_num(q_y / q, A::zero());

        let (scored_x, scored_y) = match w {
            Winner::X => (weight * win_weight, A::zero()),
            Winner::Y => (A::zero(), weight * win_weight),
            Winner::Draw => (weight * tie_weight, weight * tie_weight),
        };

        scores[x] += k * (scored_x - expected_x);
        scores[y] += k * (scored_y - expected_y);
    }

    nan_to_num(&mut scores, A::zero());

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
        let winners = array![Winner::X, Winner::Draw, Winner::Y, Winner::X];
        let weights = array![1.0, 1.0, 1.0, 1.0];
        let initial: f64 = 1500.0;
        let base: f64 = 10.0;
        let scale: f64 = 400.0;
        let k: f64 = 30.0;

        let expected = array![1501.0, 1485.0, 1515.0, 1498.0];

        let actual = elo(
            &xs.view(),
            &ys.view(),
            &winners.view(),
            &weights.view(),
            5,
            initial,
            base,
            scale,
            k,
            1.0,
            0.5,
        )
        .unwrap();

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-0, "a = {}, b = {}", a, b);
        }
    }
}
