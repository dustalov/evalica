use ndarray::{Array1, ArrayView1};

use crate::Winner;

pub fn counting(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
) -> Array1<i64> {
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
        return Array1::zeros(0);
    }

    let n = 1 + std::cmp::max(*xs.iter().max().unwrap(), *ys.iter().max().unwrap());

    let mut scores = Array1::<i64>::ones(n);

    for i in 0..xs.len() {
        match ws[i] {
            Winner::X => scores[xs[i]] += 1,
            Winner::Y => scores[ys[i]] += 1,
            _ => {}
        }
    }

    scores
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::Winner;

    use super::counting;

    #[test]
    fn test_counting() {
        let xs = array![3, 2, 1, 0];
        let ys = array![0, 1, 2, 3];
        let ws = array![Winner::X, Winner::Draw, Winner::Y, Winner::X];

        let expected = array![2, 1, 2, 2];

        let actual = counting(&xs.view(), &ys.view(), &ws.view());

        assert_eq!(expected, actual);
    }
}
