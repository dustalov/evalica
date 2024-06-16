use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand::prelude::StdRng;

fn compute_ties_and_wins(m: &Array2<i64>) -> (Array2<i64>, Array2<i64>) {
    let mut t = m.clone();
    for ((i, j), t) in t.indexed_iter_mut() {
        *t = std::cmp::min(m[[i, j]], m[[j, i]]);
    }
    let w = m - &t;
    (t, w)
}

pub fn newman(m: &Array2<i64>, seed: u64, tolerance: f64, limit: usize) -> (Array1<f64>, usize) {
    let (t, w) = compute_ties_and_wins(m);

    let mut rng = StdRng::seed_from_u64(seed);

    let mut pi: Array1<f64> = Array1::from_shape_fn(m.shape()[0], |_| rng.gen_range(0.0..1.0));
    let mut v: f64 = rng.gen_range(0.0..1.0);

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        iterations += 1;

        let pi_broadcast = pi.broadcast((pi.len(), pi.len())).unwrap().to_owned();
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

        let pi_old = pi.clone();

        let pi_numerator = ((w.mapv(|x| x as f64) + t.mapv(|x| x as f64) / 2.0)
            * (&pi_broadcast + v * &sqrt_pi_product)
            / (&pi_sum + 2.0 + v * &sqrt_pi_product))
            .sum_axis(Axis(1));

        let pi_denominator = ((w.mapv(|x| x as f64) + t.mapv(|x| x as f64) / 2.0)
            * (1.0 + v * &sqrt_pi_product)
            / (&pi_sum + 2.0 + v * &sqrt_pi_product))
            .sum_axis(Axis(0));

        pi = &pi_numerator / &pi_denominator;

        pi.iter_mut().for_each(|x| {
            if x.is_nan() {
                *x = tolerance;
            }
        });

        converged = pi.iter().zip(pi_old.iter()).all(|(p_new, p_old)| {
            (p_new / (p_new + 1.0) - p_old / (p_old + 1.0)).abs() <= tolerance
        });
    }

    (pi, iterations)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::newman;

    #[test]
    fn test_newman() {
        let m = array![
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0]
        ];

        let seed = 0;
        let tolerance = 1e-6;
        let limit = 100;

        let (pi, iterations) = newman(&m, seed, tolerance, limit);

        assert_eq!(pi.len(), m.shape()[0]);
        assert_ne!(iterations, 0);
    }
}
