use ndarray::{Array1, ArrayView1, ErrorKind, ShapeError};

use crate::{match_lengths, utils::one_nan_to_num, Winner};

pub fn elo(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    initial: f64,
    base: f64,
    scale: f64,
    k: f64,
) -> Result<Array1<f64>, ShapeError> {
    match_lengths!(xs.len(), ys.len(), ws.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let n = 1 + std::cmp::max(*xs.iter().max().unwrap(), *ys.iter().max().unwrap());

    let mut scores = Array1::<f64>::ones(n) * initial;

    for ((x, y), &ref w) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        let q_x = one_nan_to_num(base.powf(scores[*x] / scale), 0.0);
        let q_y = one_nan_to_num(base.powf(scores[*y] / scale), 0.0);

        let q = one_nan_to_num(q_x + q_y, 0.0);

        let expected_x = one_nan_to_num(q_x / q, 0.0);
        let expected_y = one_nan_to_num(q_y / q, 0.0);

        let (scored_x, scored_y) = match w {
            Winner::X => (1.0, 0.0),
            Winner::Y => (0.0, 1.0),
            Winner::Draw => (0.5, 0.5),
            _ => continue,
        };

        scores[*x] += k * (scored_x - expected_x);
        scores[*y] += k * (scored_y - expected_y);
    }

    Ok(scores)
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

        let actual = elo(&xs.view(), &ys.view(), &ws.view(), initial, base, scale, k).unwrap();

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-0, "a = {}, b = {}", a, b);
        }
    }
}
