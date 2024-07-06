use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::Winner;

pub fn compute_ties_and_wins(m: &ArrayView2<i64>) -> (Array2<i64>, Array2<i64>) {
    let mut t = m.to_owned();

    for ((i, j), t) in t.indexed_iter_mut() {
        *t = std::cmp::min(m[[i, j]], m[[j, i]]);
    }

    let w = m - &t;

    (t, w)
}

pub fn matrices(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
) -> (Array2<i64>, Array2<i64>) {
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

    assert!(!xs.is_empty(), "empty inputs");

    let n = 1 + std::cmp::max(*xs.iter().max().unwrap(), *ys.iter().max().unwrap());

    let mut wins = Array2::zeros((n, n));
    let mut ties = Array2::zeros((n, n));

    for i in 0..xs.len() {
        match ws[i] {
            Winner::X => {
                wins[[xs[i], ys[i]]] += 1;
            }
            Winner::Y => {
                wins[[ys[i], xs[i]]] += 1;
            }
            Winner::Draw => {
                ties[[xs[i], ys[i]]] += 1;
                ties[[ys[i], xs[i]]] += 1;
            }
            _ => {}
        }
    }

    (wins, ties)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{matrices, Winner};

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

        let (wins, ties) = matrices(&xs.view(), &ys.view(), &ws.view());

        assert_eq!(wins, expected_wins);

        assert_eq!(ties, expected_ties);
    }
}
