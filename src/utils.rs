use std::collections::HashMap;
use std::hash::Hash;
use std::num::FpCategory;
use std::ops::{AddAssign, MulAssign};

use ndarray::{
    Array, Array2, ArrayView1, ArrayView2, Dimension, ErrorKind, ScalarOperand, ShapeError,
};
use num_traits::{Float, Num};

use crate::Winner;

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

#[macro_export]
macro_rules! check_total {
    ($total:expr, $($xs:expr),+ $(,)?) => {{
        let mut max_value = 0;

        $(
            let iter_max = $xs.iter().max().unwrap();
            if *iter_max > max_value {
                max_value = *iter_max;
            }
        )+

        if max_value > $total {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
    }};
}

#[allow(dead_code)]
pub fn index<I: Eq + Hash + Clone>(xs: &ArrayView1<I>, ys: &ArrayView1<I>) -> HashMap<I, usize> {
    let mut index: HashMap<I, usize> = HashMap::new();

    for x in xs.iter().chain(ys.iter()) {
        let len = index.len();
        index.entry(x.clone()).or_insert(len);
    }

    index
}

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
        FpCategory::Zero => x,
        FpCategory::Subnormal => x,
        FpCategory::Normal => x,
    }
}

pub fn nan_to_num<A: Float, D: Dimension>(xs: &mut Array<A, D>, nan: A) {
    xs.map_inplace(|x| *x = one_nan_to_num(*x, nan));
}

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

pub fn matrices<A: Num + Copy + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    winners: &ArrayView1<Winner>,
    weights: &ArrayView1<A>,
    total: usize,
) -> Result<(Array2<A>, Array2<A>), ShapeError> {
    check_lengths!(xs.len(), ys.len(), winners.len(), weights.len());

    if xs.is_empty() {
        return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
    }

    check_total!(total, xs, ys);

    let mut wins = Array2::zeros((total, total));
    let mut ties = wins.clone();

    for (((&x, &y), &ref w), &weight) in xs.iter().zip(ys.iter()).zip(winners.iter()).zip(weights) {
        match w {
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

pub fn win_plus_tie_matrix<A: Float + MulAssign + ScalarOperand>(
    win_matrix: &ArrayView2<A>,
    tie_matrix: &ArrayView2<A>,
    win_weight: A,
    tie_weight: A,
    nan: A,
) -> Array2<A> {
    let mut win_matrix = win_matrix.to_owned();
    nan_to_num(&mut win_matrix, nan);
    win_matrix *= win_weight;

    let mut tie_matrix = tie_matrix.to_owned();
    nan_to_num(&mut tie_matrix, nan);
    tie_matrix *= tie_weight;

    let mut matrix = &win_matrix + &tie_matrix;
    nan_to_num(&mut matrix, nan);

    matrix
}

pub fn pairwise_scores<A: Float>(scores: &ArrayView1<A>) -> Array2<A> {
    if scores.is_empty() {
        return Array2::zeros((0, 0));
    }

    let len = scores.len();

    let mut pairwise = Array2::zeros((len, len));

    for ((i, j), value) in pairwise.indexed_iter_mut() {
        *value = scores[i] / (scores[i] + scores[j]);
    }

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
            .cloned()
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
