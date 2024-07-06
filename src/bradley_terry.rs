use ndarray::{Array1, Array2, ArrayView2, Axis};

pub fn bradley_terry(m: &ArrayView2<f64>, tolerance: f64, limit: usize) -> (Array1<f64>, usize) {
    assert_eq!(m.shape()[0], m.shape()[1], "The matrix must be square");

    let totals = m.t().to_owned() + m;

    let active = totals.mapv(|x| x > 0.0);

    let w: Array1<f64> = m.sum_axis(Axis(1));

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
                scores_new[i] = w[i] / d;
            }
        }

        let p_sum = scores_new.sum();

        if p_sum == 0.0 {
            scores_new.assign(&scores)
        } else {
            scores_new /= p_sum;
        }

        let difference = &scores_new - &scores;
        converged = difference.dot(&difference).sqrt() < tolerance;

        scores.assign(&scores_new);
    }

    (scores, iterations)
}

pub fn newman(
    w: &ArrayView2<f64>,
    t: &ArrayView2<f64>,
    v_init: f64,
    tolerance: f64,
    limit: usize,
) -> (Array1<f64>, f64, usize) {
    assert!(v_init.is_normal());
    assert!(v_init > 0.0);

    let mut scores = Array1::ones(w.shape()[0]);
    let mut scores_new = scores.clone();

    let w_t_half = w + &(t / 2.0);

    let mut converged = false;
    let mut iterations = 0;

    let mut v = v_init;

    while !converged && iterations < limit {
        iterations += 1;

        let sqrt_scores = scores.mapv(f64::sqrt);
        let w_denom = &scores + &scores.t() + 2.0 * v * &sqrt_scores;
        let w_scores = &w_t_half * (&scores + v * &sqrt_scores) / &w_denom;

        let pi_numerator = w_scores.sum_axis(Axis(1));
        let pi_denominator = w_scores.sum_axis(Axis(0));

        let t_denom = &scores + &scores.t() + 2.0 * v * &sqrt_scores;
        let v_numerator = (*&t * (&scores + &scores.t()) / t_denom).sum() / 2.0;

        let v_denom = (2.0 * w * &sqrt_scores / &w_denom).sum();

        v = v_numerator / v_denom;

        if !v.is_finite() {
            v = 0.0;
        }

        let result = &pi_numerator / &pi_denominator;

        if result.iter().all(|x| x.is_finite()) {
            scores_new.assign(&result)
        }

        let difference = &scores_new - &scores;
        converged = difference.dot(&difference).sqrt() < tolerance;
        scores.assign(&scores_new);
    }

    (scores, v, iterations)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use crate::utils;

    use super::{bradley_terry, newman};

    fn matrix() -> Array2<f64> {
        return array![
            [0.0, 1.0, 2.0, 0.0, 1.0],
            [2.0, 0.0, 2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0, 2.0],
            [2.0, 0.0, 1.0, 3.0, 0.0]
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
        let tolerance = 1e-8;
        let limit = 100;

        let (w, t) = utils::compute_ties_and_wins(&m.view());

        let w64 = w.map(|x| *x as f64);
        let t64 = t.map(|x| *x as f64);

        let (actual, v, iterations) = newman(&w64.view(), &t64.view(), 0.5, tolerance, limit);

        assert_eq!(actual.len(), m.shape()[0]);
        assert!(v.is_finite());
        assert_ne!(iterations, 0);
    }
}
