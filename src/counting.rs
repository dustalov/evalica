use ndarray::{Array1, ArrayView1};

use crate::Winner;

pub fn counting(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    win_weight: f64,
    tie_weight: f64,
) -> Array1<f64> {
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

    let mut scores = Array1::zeros(n);

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

        let expected = array![1.0, 0.5, 1.5, 1.0];

        let actual = counting(&xs.view(), &ys.view(), &ws.view(), 1.0, 0.5);

        assert_eq!(expected, actual);
    }
}
