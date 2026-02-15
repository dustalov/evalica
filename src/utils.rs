use std::collections::HashMap;
use std::hash::Hash;
use std::num::FpCategory;
use std::ops::{AddAssign, MulAssign};

use ndarray::{
    Array, Array2, ArrayView1, ArrayView2, Axis, Dimension, ErrorKind, ScalarOperand, ShapeError,
    Zip,
};
use num_traits::{Float, Num};

use crate::Winner;

/// Checks if all the given arrays have the same length.
#[macro_export]
macro_rules! check_lengths {
    ($first:expr, $($rest:expr),+) => {
        $(
            if $first != $rest {
                return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
            }
        )*
    };
}

/// Checks if the total number of items is consistent with the maximum value in the given arrays.
#[macro_export]
macro_rules! check_total {
    ($total:expr, $($xs:expr),+ $(,)?) => {{
        let max_value = std::iter::empty()
            $(
                .chain($xs.iter())
            )+
            .max()
            .map(|&x| x)
            .unwrap_or(0);

        if max_value >= $total {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
    }};
}

/// Creates a mapping from unique items to integer indices.
///
/// # Arguments
///
/// * `xs` - A 1D array of the first items in each comparison.
/// * `ys` - A 1D array of the second items in each comparison.
///
/// # Returns
///
/// A `HashMap` where keys are the unique items and values are their assigned indices.
#[allow(dead_code)]
#[must_use]
pub fn index<I: Eq + Hash + Clone>(xs: &ArrayView1<I>, ys: &ArrayView1<I>) -> HashMap<I, usize> {
    let mut index: HashMap<I, usize> = HashMap::new();

    for x in xs.iter().chain(ys.iter()) {
        let len = index.len();
        index.entry(x.clone()).or_insert(len);
    }

    index
}

/// Replaces NaN values with a specified number, and infinite values with the max/min value of the type.
///
/// # Arguments
///
/// * `x` - The float value to check.
/// * `nan` - The value to replace NaN with.
///
/// # Returns
///
/// The original value if it's not NaN or infinite, otherwise the replacement value.
pub fn one_nan_to_num<A: Float>(x: A, nan: A) -> A {
    match x.classify() {
        FpCategory::Nan => nan,
        FpCategory::Infinite => {
            if x.is_sign_positive() {
                A::max_value()
            } else {
                A::min_value()
            }
        }
        FpCategory::Zero | FpCategory::Subnormal | FpCategory::Normal => x,
    }
}

/// Replaces NaN values in an array with a specified number.
///
/// # Arguments
///
/// * `xs` - The array to modify in-place.
/// * `nan` - The value to replace NaN with.
pub fn nan_to_num<A: Float, D: Dimension>(xs: &mut Array<A, D>, nan: A) {
    xs.map_inplace(|x| *x = one_nan_to_num(*x, nan));
}

/// Calculates the mean of an array, ignoring NaN values.
///
/// # Arguments
///
/// * `xs` - The array to calculate the mean of.
///
/// # Returns
///
/// The mean of the array, ignoring NaN values.
#[must_use]
pub fn nan_mean<A: Float + AddAssign>(xs: &ArrayView1<A>) -> A {
    let mut sum = A::zero();
    let mut count = A::zero();

    for &value in xs.iter() {
        if !value.is_nan() {
            sum += value;
            count += A::one();
        }
    }

    if count > A::zero() {
        sum / count
    } else {
        A::zero()
    }
}

/// Creates the win and tie matrices from the pairwise comparison data.
///
/// # Arguments
///
/// * `xs` - A 1D array of the first items in each comparison.
/// * `ys` - A 1D array of the second items in each comparison.
/// * `winners` - A 1D array of the winners of each comparison.
/// * `weights` - A 1D array of weights for each comparison.
/// * `total` - The total number of items.
///
/// # Returns
///
/// A tuple containing:
/// * The win matrix.
/// * The tie matrix.
///
/// # Errors
///
/// Returns an error if the input arrays have different lengths or indices exceed `total`.
pub fn matrices<A, W>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    winners: &ArrayView1<W>,
    weights: &ArrayView1<A>,
    total: usize,
) -> Result<(Array2<A>, Array2<A>), ShapeError>
where
    A: Num + Copy + AddAssign,
    W: Copy + TryInto<Winner>,
{
    check_lengths!(xs.len(), ys.len(), winners.len(), weights.len());

    if xs.is_empty() {
        return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
    }

    check_total!(total, xs, ys);

    let mut wins = Array2::zeros((total, total));
    let mut ties = Array2::zeros((total, total));

    for (((&x, &y), &w), &weight) in xs.iter().zip(ys.iter()).zip(winners.iter()).zip(weights) {
        let winner = w
            .try_into()
            .map_err(|_| ShapeError::from_kind(ErrorKind::IncompatibleShape))?;
        match winner {
            Winner::X => {
                wins[[x, y]] += weight;
            }
            Winner::Y => {
                wins[[y, x]] += weight;
            }
            Winner::Draw => {
                ties[[x, y]] += weight;
                ties[[y, x]] += weight;
            }
        }
    }

    Ok((wins, ties))
}

/// Creates a combined win+tie matrix from separate win and tie matrices.
///
/// # Arguments
///
/// * `win_matrix` - The matrix of wins.
/// * `tie_matrix` - The matrix of ties.
/// * `win_weight` - The weight to apply to wins.
/// * `tie_weight` - The weight to apply to ties.
/// * `nan` - The value to replace NaN with.
///
/// # Returns
///
/// A combined win+tie matrix.
pub fn win_plus_tie_matrix<A: Float + MulAssign + ScalarOperand>(
    win_matrix: &ArrayView2<A>,
    tie_matrix: &ArrayView2<A>,
    win_weight: A,
    tie_weight: A,
    nan: A,
) -> Array2<A> {
    let mut matrix = Array2::zeros(win_matrix.raw_dim());

    Zip::from(&mut matrix)
        .and(win_matrix)
        .and(tie_matrix)
        .for_each(|r, &w, &t| {
            let val = one_nan_to_num(w, nan) * win_weight + one_nan_to_num(t, nan) * tie_weight;
            *r = one_nan_to_num(val, nan);
        });
    matrix
}

/// Calculates the pairwise scores from a 1D array of scores.
///
/// The pairwise score between item `i` and item `j` is the probability that item `i` wins against item `j`.
///
/// # Arguments
///
/// * `scores` - A 1D array of scores for each item.
///
/// # Returns
///
/// A 2D array of pairwise scores.
#[must_use]
pub fn pairwise_scores<A: Float>(scores: &ArrayView1<A>) -> Array2<A> {
    if scores.is_empty() {
        return Array2::zeros((0, 0));
    }

    let scores_col = scores.view().insert_axis(Axis(1));
    let scores_row = scores.view().insert_axis(Axis(0));

    let mut pairwise = &scores_col / (&scores_col + &scores_row);

    nan_to_num(&mut pairwise, A::zero());

    pairwise
}

#[cfg(test)]
pub mod fixtures {
    use crate::Winner;

    pub(crate) static XS: [usize; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4];
    pub(crate) static YS: [usize; 16] = [1, 2, 2, 4, 0, 2, 2, 3, 4, 0, 1, 2, 4, 4, 0, 3];
    pub(crate) static WINNERS: [Winner; 16] = [
        Winner::Draw,
        Winner::Y,
        Winner::Draw,
        Winner::Draw,
        Winner::X,
        Winner::Draw,
        Winner::Draw,
        Winner::Draw,
        Winner::Draw,
        Winner::X,
        Winner::X,
        Winner::X,
        Winner::Draw,
        Winner::Draw,
        Winner::X,
        Winner::X,
    ];
    pub(crate) static WEIGHTS: [f64; 16] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    pub(crate) static TOTAL: usize = 5;
}

#[cfg(test)]
mod tests {
    use super::{index, matrices, pairwise_scores, Winner};
    use ndarray::array;

    #[test]
    fn test_index() {
        let xs = array![0, 1, 2, 3];
        let ys = array![1, 2, 3, 4];

        let expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
            .iter()
            .copied()
            .collect();

        let actual = index(&xs.view(), &ys.view());

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_matrices() {
        let xs = array![0, 1, 2];
        let ys = array![1, 2, 3];
        let winners = array![Winner::X, Winner::Y, Winner::Draw];
        let weights = array![1, 1, 1];

        let expected_wins = array![
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ];

        let expected_ties = array![
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ];

        let (wins, ties) =
            matrices(&xs.view(), &ys.view(), &winners.view(), &weights.view(), 5).unwrap();

        assert_eq!(wins, expected_wins);
        assert_eq!(ties, expected_ties);
    }

    #[test]
    fn test_pairwise_scores() {
        let scores = array![0.0, 1.0, 3.0];

        let expected = array![[0.00, 0.00, 0.00], [1.00, 0.50, 0.25], [1.00, 0.75, 0.50]];

        let actual = pairwise_scores(&scores.view());

        assert_eq!(actual, expected);
    }
}
