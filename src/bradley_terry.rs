use std::ops::AddAssign;

use ndarray::{Array1, ArrayView2, ErrorKind, ScalarOperand, ShapeError, Zip};
use num_traits::{Float, FromPrimitive};

use crate::utils::{one_nan_to_num, win_plus_tie_matrix};

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
///
/// # Errors
///
/// Returns an error if the matrix is not square or if numeric conversion fails.
pub fn bradley_terry<A: Float + FromPrimitive + ScalarOperand + AddAssign>(
    matrix: &ArrayView2<A>,
    tolerance: A,
    limit: usize,
) -> Result<(Array1<A>, usize), ShapeError> {
    if !matrix.is_square() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let n = matrix.nrows();
    let n_as_a = A::from(n).ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleShape))?;
    let mut scores = Array1::ones(n);

    let mut converged = false;
    let mut iterations = 0;

    if matrix.is_empty() {
        return Ok((scores, iterations));
    }

    let matrix_t = matrix.t();
    let matrix_t = matrix_t.as_standard_layout();
    let mut scores_new = Array1::zeros(n);
    let mut logs = Array1::zeros(n);
    let mut diff = Array1::zeros(n);

    while !converged && iterations < limit {
        iterations += 1;

        scores_new.assign(&scores);

        for i in 0..n {
            let mut numerator = A::zero();
            let mut denominator = A::zero();
            let score_i = scores_new[i];

            for j in 0..n {
                let sum_scores = score_i + scores_new[j];
                numerator += matrix[[i, j]] * scores_new[j] / sum_scores;
                denominator += matrix_t[[i, j]] / sum_scores;
            }

            scores_new[i] = numerator / denominator;
        }

        Zip::from(&mut logs)
            .and(&scores_new)
            .for_each(|l, &x| *l = x.ln());
        let geometric_mean = (logs.sum() / n_as_a).exp();

        Zip::from(&mut diff)
            .and(&mut scores_new)
            .and(&scores)
            .for_each(|d, new, &old| {
                *new = one_nan_to_num(*new / geometric_mean, tolerance);
                *d = *new - old;
            });
        converged = diff.dot(&diff).sqrt() < tolerance;

        std::mem::swap(&mut scores, &mut scores_new);
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
///
/// # Errors
///
/// Returns an error if input matrices have different shapes or are not square.
#[allow(clippy::suboptimal_flops)]
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

    let n = win_matrix.nrows();
    let mut scores = Array1::<f64>::ones(n);
    let mut v = v_init;

    let mut converged = false;
    let mut iterations = 0;

    if win_matrix.is_empty() && tie_matrix.is_empty() {
        return Ok((scores, v, iterations));
    }

    let win_tie_half = win_plus_tie_matrix(win_matrix, tie_matrix, 1.0, 0.5, tolerance);
    let win_tie_half_t = win_tie_half.t();
    let win_tie_half_t = win_tie_half_t.as_standard_layout();

    let mut scores_new = Array1::<f64>::zeros(n);
    let mut v_new = v;

    let mut numerator_terms = Array1::<f64>::zeros(n);
    let mut denominator_terms = Array1::<f64>::zeros(n);
    let mut v_numerator_terms = ndarray::Array2::<f64>::zeros((n, n));
    let mut v_denominator_terms = ndarray::Array2::<f64>::zeros((n, n));

    while !converged && iterations < limit {
        iterations += 1;

        v = one_nan_to_num(v_new, tolerance);

        let mut squared_difference = 0.0;

        for i in 0..n {
            let score_i = scores[i];

            for j in 0..n {
                let score_j = scores[j];
                let sqrt_scores_outer = (score_j * score_i).sqrt();
                let sum_scores = score_j + score_i;
                let common_denominator = sum_scores + 2.0 * v * sqrt_scores_outer;

                numerator_terms[j] =
                    win_tie_half[[i, j]] * (score_j + v * sqrt_scores_outer) / common_denominator;
                denominator_terms[j] = win_tie_half_t[[i, j]]
                    * (1.0 + v * (score_j / score_i).sqrt())
                    / common_denominator;
                v_numerator_terms[[i, j]] = tie_matrix[[i, j]] * sum_scores / common_denominator;
                v_denominator_terms[[i, j]] =
                    win_matrix[[i, j]] * sqrt_scores_outer / common_denominator;
            }

            scores_new[i] =
                one_nan_to_num(numerator_terms.sum() / denominator_terms.sum(), tolerance);

            let difference = scores_new[i] - score_i;
            squared_difference += difference * difference;
        }

        let v_numerator = v_numerator_terms.sum();
        let v_denominator = v_denominator_terms.sum();

        v_new = (v_numerator / 2.0) / (v_denominator * 2.0);
        converged = squared_difference.sqrt() < tolerance;

        std::mem::swap(&mut scores, &mut scores_new);
    }

    if v_new.is_finite() {
        v = v_new;
    }

    Ok((scores, v, iterations))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{ArrayView1, array};

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

        assert!((v - v_init).abs() > tolerance);
        assert!(v.is_normal());
        assert_abs_diff_eq!(v, expected_v, epsilon = tolerance * 1e3);
    }
}
