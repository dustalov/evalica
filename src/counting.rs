use std::ops::{AddAssign, MulAssign};

use ndarray::{Array1, ArrayView1, Axis, ErrorKind, ScalarOperand, ShapeError};
use num_traits::Float;

use crate::utils::{matrices, nan_mean, nan_to_num, win_plus_tie_matrix};
use crate::{check_lengths, check_total, Winner};

pub fn counting<A: Float + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    winners: &ArrayView1<Winner>,
    weights: &ArrayView1<A>,
    total: usize,
    win_weight: A,
    tie_weight: A,
) -> Result<Array1<A>, ShapeError> {
    check_lengths!(xs.len(), ys.len(), winners.len(), weights.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    check_total!(total, xs, ys);

    let mut scores = Array1::zeros(total);

    for (((&x, &y), &ref w), &weight) in xs.iter().zip(ys.iter()).zip(winners.iter()).zip(weights) {
        match w {
            Winner::X => scores[x] += weight * win_weight,
            Winner::Y => scores[y] += weight * win_weight,
            Winner::Draw => {
                scores[x] += weight * tie_weight;
                scores[y] += weight * tie_weight;
            }
        }
    }

    nan_to_num(&mut scores, A::zero());

    Ok(scores)
}

pub fn average_win_rate<A: Float + AddAssign + MulAssign + ScalarOperand>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    winners: &ArrayView1<Winner>,
    weights: &ArrayView1<A>,
    total: usize,
    win_weight: A,
    tie_weight: A,
) -> Result<Array1<A>, ShapeError> {
    check_lengths!(xs.len(), ys.len(), winners.len(), weights.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let (win_matrix, tie_matrix) = matrices(xs, ys, winners, weights, total).unwrap();

    let mut matrix = win_plus_tie_matrix(
        &win_matrix.view(),
        &tie_matrix.view(),
        win_weight,
        tie_weight,
        A::zero(),
    );

    let mut denominator = &matrix + &matrix.t();
    nan_to_num(&mut denominator, A::zero());

    matrix = &matrix / &denominator;

    let mut scores = Array1::zeros(matrix.shape()[0]);

    for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
        scores[i] = nan_mean(&row);
    }

    nan_to_num(&mut scores, A::zero());

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, ArrayView1};

    use crate::utils;

    use super::{average_win_rate, counting};

    #[test]
    fn test_counting() {
        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let winners = ArrayView1::from(&utils::fixtures::WINNERS);
        let weights = ArrayView1::from(&utils::fixtures::WEIGHTS);

        let expected = array![1.5, 3.0, 3.0, 4.5, 4.0];

        let actual = counting(&xs, &ys, &winners, &weights, 5, 1.0, 0.5).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_average_win_rate() {
        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let winners = ArrayView1::from(&utils::fixtures::WINNERS);
        let weights = ArrayView1::from(&utils::fixtures::WEIGHTS);

        let expected = array![0.1875, 0.5, 0.4375, 0.7708333333333334, 0.6388888888888888];

        let actual = average_win_rate(&xs, &ys, &winners, &weights, 5, 1.0, 0.5).unwrap();

        assert_eq!(actual, expected);
    }
}
