use std::ops::AddAssign;

use ndarray::{Array1, ArrayView1, Axis, ErrorKind, ShapeError};
use num_traits::{Float, Num};

use crate::{check_lengths, check_total, Winner};
use crate::utils::{matrices, nan_mean, nan_to_num};

pub fn counting<A: Num + Copy + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    total: usize,
    win_weight: A,
    tie_weight: A,
) -> Result<Array1<A>, ShapeError> {
    check_lengths!(xs.len(), ys.len(), ws.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    check_total!(xs, ys, total);

    let mut scores = Array1::zeros(total);

    for ((x, y), &ref w) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        match w {
            Winner::X => scores[*x] += win_weight,
            Winner::Y => scores[*y] += win_weight,
            Winner::Draw => {
                scores[*x] += tie_weight;
                scores[*y] += tie_weight;
            }
            _ => {}
        }
    }

    Ok(scores)
}

pub fn average_win_rate<A: Float + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    total: usize,
    win_weight: A,
    tie_weight: A,
) -> Result<Array1<A>, ShapeError> {
    check_lengths!(xs.len(), ys.len(), ws.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let (win_matrix, tie_matrix) = matrices(xs, ys, ws, total, win_weight, tie_weight).unwrap();

    let mut matrix = &win_matrix + &tie_matrix;

    let matrix_t = matrix.t();
    matrix = &matrix / (&matrix + &matrix_t);

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
        let ws = ArrayView1::from(&utils::fixtures::WS);

        let expected = array![1.5, 3.0, 3.0, 4.5, 4.0];

        let actual = counting(&xs.view(), &ys.view(), &ws.view(), 5, 1.0, 0.5).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_average_win_rate() {
        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let ws = ArrayView1::from(&utils::fixtures::WS);

        let expected = array![0.1875, 0.5, 0.4375, 0.7708333333333334, 0.6388888888888888];

        let actual = average_win_rate(&xs.view(), &ys.view(), &ws.view(), 5, 1.0, 0.5).unwrap();

        assert_eq!(actual, expected);
    }
}
