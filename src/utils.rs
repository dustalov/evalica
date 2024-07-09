use std::collections::HashMap;
use std::hash::Hash;
use std::num::FpCategory;
use std::ops::AddAssign;

use ndarray::{Array1, Array2, ArrayView1};
use num_traits::{Float, Num};

use crate::Winner;

#[allow(dead_code)]
pub fn index<I: Eq + Hash + Clone>(xs: &ArrayView1<I>, ys: &ArrayView1<I>) -> HashMap<I, usize> {
    let mut index: HashMap<I, usize> = HashMap::new();

    for x in xs.iter() {
        let len = index.len();
        index.entry(x.clone()).or_insert(len);
    }

    for y in ys.iter() {
        let len = index.len();
        index.entry(y.clone()).or_insert(len);
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

pub fn matrices<A: Num + Copy + AddAssign, B: Num + Copy + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    win_weight: A,
    tie_weight: B,
) -> (Array2<A>, Array2<B>) {
    assert_eq!(
        xs.len(),
        ys.len(),
        "first and second length mismatch: {} vs. {}",
        xs.len(),
        ys.len()
    );

    assert_eq!(
        xs.len(),
        ws.len(),
        "first and status length mismatch: {} vs. {}",
        xs.len(),
        ws.len()
    );

    if xs.is_empty() {
        return (Array2::zeros((0, 0)), Array2::zeros((0, 0)));
    }

    let n = 1 + std::cmp::max(*xs.iter().max().unwrap(), *ys.iter().max().unwrap());

    let mut wins = Array2::zeros((n, n));
    let mut ties = Array2::zeros((n, n));

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

    (wins, ties)
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

        let (wins, ties) = matrices(&xs.view(), &ys.view(), &ws.view(), 1, 1);

        assert_eq!(wins, expected_wins);
        assert_eq!(ties, expected_ties);
    }
}
