use ndarray::Array1;

use crate::utils::Status;

pub fn elo(
    first: &Vec<usize>,
    second: &Vec<usize>,
    status: &Vec<Status>,
    r: f64,
    k: u64,
    s: f64,
) -> Array1<f64> {
    let n = 1 + std::cmp::max(*first.iter().max().unwrap(), *second.iter().max().unwrap());

    let mut ratings = Array1::<f64>::ones(n) * r;

    for i in 0..first.len() {
        let rating1 = ratings[first[i]];
        let rating2 = ratings[second[i]];

        let ea = 1.0 / (1.0 + 10.0f64.powf((rating2 - rating1) / s));
        let eb = 1.0 / (1.0 + 10.0f64.powf((rating1 - rating2) / s));

        match status[i] {
            Status::Won => {
                ratings[first[i]] = rating1 + k as f64 * (1.0 - ea);
                ratings[second[i]] = rating2 + k as f64 * (0.0 - eb);
            }
            Status::Lost => {
                ratings[first[i]] = rating1 + k as f64 * (0.0 - ea);
                ratings[second[i]] = rating2 + k as f64 * (1.0 - eb);
            }
            Status::Tied => {
                ratings[first[i]] = rating1 + k as f64 * (0.5 - ea);
                ratings[second[i]] = rating2 + k as f64 * (0.5 - eb);
            }
            _ => {}
        }
    }

    ratings
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_elo() {
        let first = vec![3, 2, 1, 0];
        let second = vec![0, 1, 2, 3];
        let status = vec![Status::Won, Status::Tied, Status::Lost, Status::Won];
        let r: f64 = 1500.0;
        let k: u64 = 30;
        let s: f64 = 400.0;

        let expected = array![1501.0, 1485.0, 1515.0, 1498.0];
        let actual = elo(&first, &second, &status, r, k, s);

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-0, "a = {}, b = {}", a, b);
        }
    }
}
