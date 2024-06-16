use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Norm;

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

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::{bradley_terry, EPS};

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
}
