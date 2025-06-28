use std::ops::{AddAssign, DivAssign};

use ndarray::{Array1, ArrayView2, Axis, ErrorKind, ScalarOperand, ShapeError};
use num_traits::{Float, FromPrimitive};

use crate::utils::{nan_to_num, one_nan_to_num, win_plus_tie_matrix};

/// Implements the Bradley–Terry model for pairwise comparisons.
///
/// The Bradley–Terry model is a probabilistic model that predicts the outcome of a pairwise comparison.
/// It assigns a score to each item, and the probability of item `i` winning against item `j` is
/// proportional to the ratio of their scores.
///
/// # Arguments
///
/// * `matrix` - A 2D array representing the pairwise comparison matrix. `matrix[i][j]` is the number of times item `i` won against item `j`.
/// * `tolerance` - The tolerance for checking convergence.
/// * `limit` - The maximum number of iterations.
///
/// # Returns
///
/// A tuple containing:
/// * A 1D array of scores for each item.
/// * The number of iterations performed.
pub fn bradley_terry<A: Float + FromPrimitive + ScalarOperand + AddAssign + DivAssign>(
    matrix: &ArrayView2<A>,
    tolerance: A,
    limit: usize,
) -> Result<(Array1<A>, usize), ShapeError> {
    if !matrix.is_square() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let mut scores = Array1::ones(matrix.shape()[0]);

    let mut converged = false;
    let mut iterations = 0;

    if matrix.is_empty() {
        return Ok((scores, iterations));
    }

    while !converged && iterations < limit {
        iterations += 1;

        let mut scores_new = scores.clone();

        for i in 0..matrix.nrows() {
            let mut numerator = A::zero();
            let mut denominator = A::zero();

            for j in 0..matrix.ncols() {
                let sum_scores = scores_new[i] + scores_new[j];
                numerator += matrix[[i, j]] * scores_new[j] / sum_scores;
                denominator += matrix[[j, i]] / sum_scores;
            }

            scores_new[i] = numerator / denominator;
        }

        let geometric_mean = scores_new.mapv(|x| x.ln()).mean().unwrap().exp();
        scores_new /= geometric_mean;

        nan_to_num(&mut scores_new, tolerance);

        let difference = &scores_new - &scores;
        converged = difference.dot(&difference).sqrt() < tolerance;

        scores.assign(&scores_new);
    }

    Ok((scores, iterations))
}

/// Implements the Newman model for pairwise comparisons.
///
/// The Newman model is an extension of the Bradley–Terry model that accounts for ties.
/// It introduces a parameter `v` that represents the probability of a draw.
///
/// # Arguments
///
/// * `win_matrix` - A 2D array representing the pairwise win matrix. `win_matrix[i][j]` is the number of times item `i` won against item `j`.
/// * `tie_matrix` - A 2D array representing the pairwise tie matrix. `tie_matrix[i][j]` is the number of times item `i` tied with item `j`.
/// * `v_init` - The initial value for the `v` parameter.
/// * `tolerance` - The tolerance for checking convergence.
/// * `limit` - The maximum number of iterations.
///
/// # Returns
///
/// A tuple containing:
/// * A 1D array of scores for each item.
/// * The final value of the `v` parameter.
/// * The number of iterations performed.
pub fn newman(
    win_matrix: &ArrayView2<f64>,
    tie_matrix: &ArrayView2<f64>,
    v_init: f64,
    tolerance: f64,
    limit: usize,
) -> Result<(Array1<f64>, f64, usize), ShapeError> {
    if win_matrix.shape() != tie_matrix.shape() || !win_matrix.is_square() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let mut scores = Array1::<f64>::ones(win_matrix.shape()[0]);
    let mut v = v_init;

    let mut converged = false;
    let mut iterations = 0;

    if win_matrix.is_empty() && tie_matrix.is_empty() {
        return Ok((scores, v, iterations));
    }

    let win_tie_half = win_plus_tie_matrix(&win_matrix, &tie_matrix, 1.0, 0.5, tolerance);

    let mut v_new = v;

    while !converged && iterations < limit {
        iterations += 1;

        v = one_nan_to_num(v_new, tolerance);

        let broadcast_scores_t = scores
            .clone()
            .into_shape_with_order((1, scores.len()))
            .unwrap();
        let sqrt_scores_outer =
            (&broadcast_scores_t * &broadcast_scores_t.t()).mapv_into(f64::sqrt);
        let sum_scores = &broadcast_scores_t + &broadcast_scores_t.t();
        let sqrt_div_scores_outer_t =
            (&broadcast_scores_t / &broadcast_scores_t.t()).mapv_into(f64::sqrt);
        let common_denominator = &sum_scores + 2.0 * v * &sqrt_scores_outer;

        let scores_numerator = (&win_tie_half * (&broadcast_scores_t + v * &sqrt_scores_outer)
            / &common_denominator)
            .sum_axis(Axis(1));

        let scores_denominator = (&win_tie_half.t() * (1.0 + v * &sqrt_div_scores_outer_t)
            / &common_denominator)
            .sum_axis(Axis(1));

        let mut scores_new = &scores_numerator / &scores_denominator;
        nan_to_num(&mut scores_new, tolerance);

        let v_numerator = (tie_matrix * &sum_scores / &common_denominator).sum() / 2.0;
        let v_denominator = (win_matrix * &sqrt_scores_outer / &common_denominator).sum() * 2.0;
        v_new = v_numerator / v_denominator;

        let difference = &scores_new - &scores;
        converged = difference.dot(&difference).sqrt() < tolerance;

        scores.assign(&scores_new);
    }

    Ok((scores, v, iterations))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayView1};

    use crate::utils;
    use crate::utils::{matrices, win_plus_tie_matrix};

    use super::{bradley_terry, newman};

    #[test]
    fn test_bradley_terry() {
        let tolerance = 1e-8;

        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let winners = ArrayView1::from(&utils::fixtures::WINNERS);
        let weights = ArrayView1::from(&utils::fixtures::WEIGHTS);

        let (win_matrix, tie_matrix) =
            matrices(&xs, &ys, &winners, &weights, utils::fixtures::TOTAL).unwrap();

        let matrix =
            win_plus_tie_matrix(&win_matrix.view(), &tie_matrix.view(), 1.0, 0.5, tolerance);

        let expected = array![
            0.050799672530389396,
            0.1311506914535368,
            0.13168568070110523,
            0.33688868041573855,
            0.34947527489923,
        ];

        let (actual, iterations) = bradley_terry(&matrix.view(), tolerance, 100).unwrap();

        assert_eq!(actual.len(), matrix.shape()[0]);
        assert_ne!(iterations, 0);

        let actual_normalized = actual.clone() / actual.sum();

        for (left, right) in actual_normalized.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = tolerance * 1e1);
        }
    }

    #[test]
    fn test_newman() {
        let tolerance = 1e-8;

        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let winners = ArrayView1::from(&utils::fixtures::WINNERS);
        let weights = ArrayView1::from(&utils::fixtures::WEIGHTS);

        let (win_matrix, tie_matrix) =
            matrices(&xs, &ys, &winners, &weights, utils::fixtures::TOTAL).unwrap();

        let expected_v = 3.4609664512240546;
        let v_init = 0.5;

        let expected = array![
            0.01797396524323986,
            0.48024292914742794,
            0.4814219179584765,
            12.805581189559241,
            13.036548812823787,
        ];

        let (actual, v, iterations) = newman(
            &win_matrix.view(),
            &tie_matrix.view(),
            v_init,
            tolerance,
            100,
        )
        .unwrap();

        assert_eq!(actual.len(), win_matrix.shape()[0]);
        assert_eq!(actual.len(), tie_matrix.shape()[0]);
        assert_ne!(iterations, 0);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = tolerance * 1e1);
        }

        assert_ne!(v, v_init);
        assert!(v.is_normal());
        assert_abs_diff_eq!(v, expected_v, epsilon = tolerance * 1e3);
    }
}
