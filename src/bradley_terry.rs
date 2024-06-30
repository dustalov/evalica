use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::{Rng, SeedableRng};
use rand::prelude::StdRng;

use crate::utils;

pub fn bradley_terry(m: &ArrayView2<i64>, tolerance: f64, limit: usize) -> (Array1<f64>, usize) {
    assert_eq!(m.shape()[0], m.shape()[1], "The matrix must be square");

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

        let diff_norm = (&scores_new - &scores)
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();

        converged = diff_norm < tolerance;

        scores.assign(&scores_new);
    }

    (scores, iterations)
}

pub fn newman(
    m: &ArrayView2<i64>,
    seed: u64,
    tolerance: f64,
    limit: usize,
) -> (Array1<f64>, usize) {
    assert_eq!(m.shape()[0], m.shape()[1], "The matrix must be square");

    let (t, w) = utils::compute_ties_and_wins(m);

    let mut rng = StdRng::seed_from_u64(seed);

    let mut scores: Array1<f64> = Array1::from_shape_fn(m.shape()[0], |_| rng.gen_range(0.0..1.0));
    let mut v: f64 = rng.gen_range(0.0..1.0);

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        iterations += 1;

        let pi_broadcast = scores
            .broadcast((scores.len(), scores.len()))
            .unwrap()
            .to_owned();
        let pi_broadcast_t = pi_broadcast.t().to_owned();
        let pi_sum = &pi_broadcast + &pi_broadcast_t;
        let sqrt_pi_product = (pi_broadcast.clone() * pi_broadcast_t.clone()).mapv(f64::sqrt);

        let denominator_common = &pi_sum + 2.0 * v * &sqrt_pi_product;

        let v_numerator =
            (&t.mapv(|x| x as f64) * (&pi_broadcast + &pi_broadcast_t) / &denominator_common).sum()
                / 2.0;

        let v_denominator =
            (&w.mapv(|x| x as f64) * (2.0 * &sqrt_pi_product) / &denominator_common).sum();

        v = v_numerator / v_denominator;

        if v.is_nan() {
            v = tolerance;
        }

        let scores_old = scores.clone();

        let pi_numerator = ((w.mapv(|x| x as f64) + t.mapv(|x| x as f64) / 2.0)
            * (&pi_broadcast + v * &sqrt_pi_product)
            / (&pi_sum + 2.0 + v * &sqrt_pi_product))
            .sum_axis(Axis(1));

        let pi_denominator = ((w.mapv(|x| x as f64) + t.mapv(|x| x as f64) / 2.0)
            * (1.0 + v * &sqrt_pi_product)
            / (&pi_sum + 2.0 + v * &sqrt_pi_product))
            .sum_axis(Axis(0));

        scores = &pi_numerator / &pi_denominator;

        scores.iter_mut().for_each(|x| {
            if x.is_nan() {
                *x = tolerance;
            }
        });

        converged = scores.iter().zip(scores_old.iter()).all(|(p_new, p_old)| {
            (p_new / (p_new + 1.0) - p_old / (p_old + 1.0)).abs() <= tolerance
        });
    }

    (scores, iterations)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::{bradley_terry, newman};

    fn matrix() -> Array2<i64> {
        return array![
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0]
        ];
    }

    #[test]
    fn test_bradley_terry() {
        let m = matrix();
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

    #[test]
    fn test_newman() {
        let m = matrix();
        let seed = 0;
        let tolerance = 1e-8;
        let limit = 100;

        let (actual, iterations) = newman(&m.view(), seed, tolerance, limit);

        assert_eq!(actual.len(), m.shape()[0]);
        assert_ne!(iterations, 0);
    }
}
