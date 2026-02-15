use std::ops::DivAssign;

use ndarray::{Array1, Array2, ArrayView2, ErrorKind, ScalarOperand, ShapeError};
use num_traits::Float;

use crate::utils::nan_to_num;

/// Calculates the eigenvector of a matrix.
///
/// # Arguments
///
/// * `matrix` - A 2D array representing the matrix.
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
pub fn eigen<A: Float + ScalarOperand + DivAssign>(
    matrix: &ArrayView2<A>,
    tolerance: A,
    limit: usize,
) -> Result<(Array1<A>, usize), ShapeError> {
    if !matrix.is_square() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let n = matrix.shape()[0];
    let n_as_a = A::from(n).ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleShape))?;
    let matrix_t = matrix.t();

    let mut scores = Array1::from_elem(n, A::one() / n_as_a);
    let mut scores_new = scores.clone();

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        iterations += 1;

        scores_new.assign(&matrix_t.dot(&scores));

        let norm = scores_new.dot(&scores_new).sqrt();

        if !norm.is_zero() {
            scores_new /= norm;
        }

        nan_to_num(&mut scores_new, tolerance);

        let mut squared_difference = A::zero();
        for (&new_value, &old_value) in scores_new.iter().zip(scores.iter()) {
            let difference = new_value - old_value;
            squared_difference = squared_difference + difference * difference;
        }
        converged = squared_difference.sqrt() < tolerance;

        scores.assign(&scores_new);
    }

    Ok((scores, limit))
}

fn pagerank_matrix(matrix: &ArrayView2<f64>, damping: f64) -> Array2<f64> {
    if matrix.shape()[0] == 0 {
        return Array2::<f64>::zeros((0, 0));
    }

    let p = 1.0 / matrix.shape()[0] as f64;

    let mut matrix = matrix.t().to_owned();

    for mut row in matrix.outer_iter_mut() {
        let sum = row.sum();

        if sum == 0.0 {
            row.fill(p);
        } else {
            row /= sum;
        }
    }

    damping * matrix + (1.0 - damping) * p
}

/// Implements the `PageRank` algorithm.
///
/// `PageRank` is an algorithm used by `Google Search` to rank web pages in search results.
///
/// # Arguments
///
/// * `matrix` - A 2D array representing the pairwise comparison matrix.
/// * `damping` - The damping factor.
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
/// Returns an error if the input matrix is not square.
pub fn pagerank(
    matrix: &ArrayView2<f64>,
    damping: f64,
    tolerance: f64,
    limit: usize,
) -> Result<(Array1<f64>, usize), ShapeError> {
    if !matrix.is_square() {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let pagerank_matrix = pagerank_matrix(matrix, damping);

    let result = eigen(&pagerank_matrix.view(), tolerance, limit);

    match result {
        Ok((mut scores, iterations)) => {
            scores /= scores.sum();

            Ok((scores, iterations))
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{array, ArrayView1};

    use crate::utils;
    use crate::utils::{matrices, win_plus_tie_matrix};

    use super::*;

    #[test]
    fn test_eigen() {
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
            0.6955953825629276,
            0.4177536241125358,
            0.4195816294550065,
            0.2806190746997716,
            0.2946746755941242,
        ];

        let (actual, iterations) = eigen(&matrix.view(), tolerance, 100).unwrap();

        assert_eq!(actual.len(), matrix.shape()[0]);
        assert!(iterations > 0);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(left, right, epsilon = tolerance);
        }
    }

    #[test]
    fn test_pagerank() {
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
            0.13280040999661397,
            0.15405089709854697,
            0.16240277665024724,
            0.2779483968089382,
            0.2727975194456537,
        ];

        let (actual, iterations) = pagerank(&matrix.view(), 0.85, tolerance, 100).unwrap();

        assert!(iterations > 0);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(left, right, epsilon = tolerance);
        }
    }
}
