use numpy::PyArrayMethods;
use pyo3::prelude::*;

use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_linalg::Norm;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use rand::prelude::*;

const EPS: f64 = 1e-8;

pub fn bradley_terry(m: &Array2<i64>) -> (Array1<f64>, usize) {
    let t = m.t().to_owned() + m;

    let active = t.mapv(|x| x > 0);

    let w: Array1<i64> = m.sum_axis(Axis(1));

    let mut z: Array2<f64> = Array2::zeros(m.raw_dim());

    let mut p: Array1<f64> = Array1::ones(m.shape()[0]);
    let mut p_new: Array1<f64> = p.clone();

    let mut converged = false;
    let mut iterations = 0;

    while !converged {
        iterations += 1;

        for ((i, j), &active_val) in active.indexed_iter() {
            if active_val {
                z[[i, j]] = t[[i, j]] as f64 / (p[i] + p[j]);
            }
        }

        p_new.fill(0.0);

        for i in 0..m.shape()[0] {
            p_new[i] = w[i] as f64 / z.column(i).sum();
        }

        p_new /= p_new.sum();

        let diff_norm = (&p_new - &p).norm();

        converged = diff_norm < EPS;

        p.assign(&p_new);
    }

    (p, iterations)
}

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
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bradley_terry() {
        let m: Array2<i64> = array![
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0]
        ];

        let (p, iterations) = bradley_terry(&m);

        assert_eq!(p.len(), m.shape()[0]);
        assert_ne!(iterations, 0);

        let expected_p = array![0.12151104, 0.15699947, 0.11594851, 0.31022851, 0.29531247];

        for (a, b) in p.iter().zip(expected_p.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

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

#[pyfunction]
fn py_bradley_terry(py: Python, m: &Bound<PyArray2<i64>>) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let m = unsafe { m.as_array().to_owned() };
    let (pi, iterations) = bradley_terry(&m);
    Ok((pi.into_pyarray_bound(py).unbind(), iterations))
}

#[pyfunction]
fn py_newman(
    py: Python,
    m: &Bound<PyArray2<i64>>,
    seed: u64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let m = unsafe { m.as_array().to_owned() };
    let (pi, iterations) = newman(&m, seed, tolerance, limit);
    Ok((pi.into_pyarray_bound(py).unbind(), iterations))
}

#[pymodule]
fn evalica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_bradley_terry, m)?)?;
    m.add_function(wrap_pyfunction!(py_newman, m)?)?;
    Ok(())
}
