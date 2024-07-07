use ndarray::{Array1, ArrayView1};

use crate::Winner;

pub fn elo(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    initial: f64,
    base: f64,
    scale: f64,
    k: f64,
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

    let mut scores = Array1::<f64>::ones(n) * initial;

    for ((x, y), &ref w) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        let q_x = base.powf(scores[*x] / scale);
        let q_y = base.powf(scores[*y] / scale);

        let q = q_x + q_y;

        let expected_x = q_x / q;
        let expected_y = q_y / q;

        let (scored_x, scored_y) = match w {
            Winner::X => (1.0, 0.0),
            Winner::Y => (0.0, 1.0),
            Winner::Draw => (0.5, 0.5),
            _ => (0.0, 0.0),
        };

        scores[*x] += k * (scored_x - expected_x);
        scores[*y] += k * (scored_y - expected_y);
    }

    scores
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_elo() {
        let xs = array![3, 2, 1, 0];
        let ys = array![0, 1, 2, 3];
        let ws = array![Winner::X, Winner::Draw, Winner::Y, Winner::X];
        let initial: f64 = 1500.0;
        let base: f64 = 10.0;
        let scale: f64 = 400.0;
        let k: f64 = 30.0;

        let expected = array![1501.0, 1485.0, 1515.0, 1498.0];

        let actual = elo(&xs.view(), &ys.view(), &ws.view(), initial, base, scale, k);

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-0, "a = {}, b = {}", a, b);
        }
    }
}
