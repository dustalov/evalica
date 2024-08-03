use std::collections::HashMap;
use std::hash::Hash;
use std::num::FpCategory;
use std::ops::AddAssign;

use ndarray::{Array1, Array2, ArrayView1, ErrorKind, ShapeError};
use num_traits::{Float, Num};

use crate::Winner;

#[macro_export]
macro_rules! check_lengths {
    ($xs:expr, $ys:expr, $ws:expr) => {
        if $xs != $ys || $xs != $ws || $ys != $ws {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
    };
}

#[macro_export]
macro_rules! check_total {
    ($xs:expr, $ys:expr, $total:expr) => {
        if std::cmp::max(*$xs.iter().max().unwrap(), *$ys.iter().max().unwrap()) > $total {
            return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
        }
    };
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

pub fn nan_to_num<A: Float>(xs: &mut Array1<A>, nan: A) {
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

pub fn matrices<A: Num + Copy + AddAssign, B: Num + Copy + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    total: usize,
    win_weight: A,
    tie_weight: B,
) -> Result<(Array2<A>, Array2<B>), ShapeError> {
    check_lengths!(xs.len(), ys.len(), ws.len());

    if xs.is_empty() {
        return Ok((Array2::zeros((0, 0)), Array2::zeros((0, 0))));
    }

    check_total!(xs, ys, total);

    let mut wins = Array2::zeros((total, total));
    let mut ties = Array2::zeros((total, total));

    for ((x, y), &ref w) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        match w {
            Winner::X => {
                wins[[*x, *y]] += win_weight;
            }
            Winner::Y => {
                wins[[*y, *x]] += win_weight;
            }
            Winner::Draw => {
                ties[[*x, *y]] += tie_weight;
                ties[[*y, *x]] += tie_weight;
            }
            _ => {}
        }
    }

    Ok((wins, ties))
}

#[cfg(test)]
pub mod fixtures {
    use crate::Winner;

    pub(crate) static XS: [usize; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4];
    pub(crate) static YS: [usize; 16] = [1, 2, 2, 4, 0, 2, 2, 3, 4, 0, 1, 2, 4, 4, 0, 3];
    pub(crate) static WS: [Winner; 16] = [
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
    pub(crate) static TOTAL: usize = 5;
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{index, matrices, Winner};

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
        let xs = array![0, 1, 2, 3];
        let ys = array![1, 2, 3, 4];
        let ws = array![Winner::X, Winner::Y, Winner::Draw, Winner::Ignore];

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

        let (wins, ties) = matrices(&xs.view(), &ys.view(), &ws.view(), 5, 1, 1).unwrap();

        assert_eq!(wins, expected_wins);
        assert_eq!(ties, expected_ties);
    }
}
