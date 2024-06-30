use ndarray::{Array1, ArrayView1};

use crate::Status;

pub fn elo(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    rs: &ArrayView1<Status>,
    r: f64,
    k: u64,
    s: f64,
) -> Array1<f64> {
    let n = 1 + std::cmp::max(*xs.iter().max().unwrap(), *ys.iter().max().unwrap());

    let mut scores = Array1::<f64>::ones(n) * r;

    for i in 0..xs.len() {
        let rating1 = scores[xs[i]];
        let rating2 = scores[ys[i]];

        let expected_a = 1.0 / (1.0 + 10.0f64.powf((rating2 - rating1) / s));
        let expected_b = 1.0 / (1.0 + 10.0f64.powf((rating1 - rating2) / s));

        match rs[i] {
            Status::Won => {
                scores[xs[i]] = rating1 + k as f64 * (1.0 - expected_a);
                scores[ys[i]] = rating2 + k as f64 * (0.0 - expected_b);
            }
            Status::Lost => {
                scores[xs[i]] = rating1 + k as f64 * (0.0 - expected_a);
                scores[ys[i]] = rating2 + k as f64 * (1.0 - expected_b);
            }
            Status::Tied => {
                scores[xs[i]] = rating1 + k as f64 * (0.5 - expected_a);
                scores[ys[i]] = rating2 + k as f64 * (0.5 - expected_b);
            }
            _ => {}
        }
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
        let rs = array![Status::Won, Status::Tied, Status::Lost, Status::Won];
        let r: f64 = 1500.0;
        let k: u64 = 30;
        let s: f64 = 400.0;

        let expected = array![1501.0, 1485.0, 1515.0, 1498.0];

        let actual = elo(&xs.view(), &ys.view(), &rs.view(), r, k, s);

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-0, "a = {}, b = {}", a, b);
        }
    }
}
