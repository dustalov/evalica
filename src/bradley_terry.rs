use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::Norm;

pub fn bradley_terry(m: &ArrayView2<i64>, tolerance: f64, limit: usize) -> (Array1<f64>, usize) {
    let totals = m.t().to_owned() + m;

    let active = totals.mapv(|x| x > 0);

    let w: Array1<i64> = m.sum_axis(Axis(1));

    let mut z: Array2<f64> = Array2::zeros(m.raw_dim());

    let mut scores: Array1<f64> = Array1::ones(m.shape()[0]);
    let mut scores_new: Array1<f64> = scores.clone();

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        iterations += 1;

        for ((i, j), &active_val) in active.indexed_iter() {
            if active_val {
                z[[i, j]] = totals[[i, j]] as f64 / (scores[i] + scores[j]);
            }
        }

        scores_new.fill(0.0);

        for i in 0..m.shape()[0] {
            let d = z.column(i).sum();

            if d != 0.0 {
                scores_new[i] = w[i] as f64 / d;
            }
        }

        let p_sum = scores_new.sum();

        if p_sum != 0.0 {
            scores_new /= p_sum;
        }

        let diff_norm = (&scores_new - &scores).norm();

        converged = diff_norm < tolerance;

        scores.assign(&scores_new);
    }

    (scores, iterations)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::bradley_terry;

    #[test]
    fn test_bradley_terry() {
        let m: Array2<i64> = array![
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0]
        ];

        let tolerance = 1e-8;
        let limit = 100;

        let expected = array![0.12151104, 0.15699947, 0.11594851, 0.31022851, 0.29531247];

        let (actual, iterations) = bradley_terry(&m.view(), tolerance, limit);

        assert_eq!(actual.len(), m.shape()[0]);
        assert_ne!(iterations, 0);

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < tolerance);
        }
    }
}
