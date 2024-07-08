use ndarray::{Array1, Array2, ArrayView2, Axis};

pub fn bradley_terry(
    matrix: &ArrayView2<f64>,
    tolerance: f64,
    limit: usize,
) -> (Array1<f64>, usize) {
    assert_eq!(
        matrix.shape()[0],
        matrix.shape()[1],
        "The matrix must be square"
    );

    let totals = matrix.t().to_owned() + matrix;

    let active = totals.mapv(|x| x > 0.0);

    let w: Array1<f64> = matrix.sum_axis(Axis(1));

    let mut z: Array2<f64> = Array2::zeros(matrix.raw_dim());

    let mut scores: Array1<f64> = Array1::ones(matrix.shape()[0]);
    let mut scores_new: Array1<f64> = scores.clone();

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        iterations += 1;

        for ((i, j), &active_val) in active.indexed_iter() {
            if active_val {
                z[[i, j]] = totals[[i, j]] / (scores[i] + scores[j]);
            }
        }

        for i in 0..matrix.shape()[0] {
            let d = z.column(i).sum();

            if d != 0.0 {
                scores_new[i] = w[i] / d;
            }
        }

        let p_sum = scores_new.sum();

        scores_new /= p_sum;

        scores_new.iter_mut().for_each(|x| {
            if x.is_nan() {
                *x = tolerance;
            }
        });

        let difference = &scores_new - &scores;
        converged = difference.dot(&difference).sqrt() < tolerance;

        scores.assign(&scores_new);
    }

    (scores, iterations)
}

pub fn newman(
    win_matrix: &ArrayView2<f64>,
    tie_matrix: &ArrayView2<f64>,
    v_init: f64,
    tolerance: f64,
    limit: usize,
) -> (Array1<f64>, f64, usize) {
    assert_eq!(
        win_matrix.shape(),
        tie_matrix.shape(),
        "The matrices must be have the same shape"
    );

    assert_eq!(
        win_matrix.shape()[0],
        win_matrix.shape()[1],
        "The win matrix must be square"
    );

    assert_eq!(
        tie_matrix.shape()[0],
        tie_matrix.shape()[1],
        "The tie matrix must be square"
    );

    assert!(v_init.is_normal());
    assert!(v_init > 0.0);

    let win_tie_half = win_matrix + &(tie_matrix / 2.0);

    let mut scores = Array1::<f64>::ones(win_matrix.shape()[0]);
    let mut scores_new = scores.clone();
    let mut v = v_init;
    let mut v_new = v;

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        iterations += 1;

        v = if v_new.is_nan() { tolerance } else { v_new };

        for i in 0..win_matrix.shape()[0] {
            let mut i_numerator = 0.0;
            let mut i_denominator = 0.0;

            for j in 0..win_matrix.shape()[1] {
                let sqrt_scores_ij = (scores[i] * scores[j]).sqrt();
                let ij_numerator = scores[j] + v * sqrt_scores_ij;
                let ij_denominator = scores[i] + scores[j] + 2.0 * v * sqrt_scores_ij;

                i_numerator += win_tie_half[[i, j]] * ij_numerator / ij_denominator;
            }

            for j in 0..win_matrix.shape()[1] {
                let sqrt_scores_ij = (scores[i] * scores[j]).sqrt();
                let ij_num = 1.0 + v * (scores[j] / scores[i]).sqrt();
                let ij_den = scores[i] + scores[j] + 2.0 * v * sqrt_scores_ij;

                i_denominator += win_tie_half[[j, i]] * ij_num / ij_den;
            }

            scores_new[i] = i_numerator / i_denominator;
        }

        scores_new.iter_mut().for_each(|x| {
            if x.is_nan() {
                *x = tolerance;
            }
        });

        let mut v_numerator = 0.0;
        let mut v_denominator = 0.0;

        for i in 0..win_matrix.shape()[0] {
            for j in 0..win_matrix.shape()[1] {
                let sqrt_scores_ij = (scores[i] * scores[j]).sqrt();
                v_numerator += tie_matrix[[i, j]] / 2.0 * (scores[i] + scores[j])
                    / (scores[i] + scores[j] + 2.0 * v * sqrt_scores_ij);
                v_denominator += win_matrix[[i, j]] * (2.0 * sqrt_scores_ij)
                    / (scores[i] + scores[j] + 2.0 * v * sqrt_scores_ij);
            }
        }

        v_new = v_numerator / v_denominator;

        let difference = &scores_new - &scores;
        converged = difference.dot(&difference).sqrt() < tolerance;

        scores = scores_new.clone();
    }

    (scores, v, iterations)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{array, ArrayView1};

    use crate::utils;
    use crate::utils::matrices;

    use super::{bradley_terry, newman};

    #[test]
    fn test_bradley_terry() {
        let tolerance = 1e-8;

        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let ws = ArrayView1::from(&utils::fixtures::WS);

        let (win_matrix, tie_matrix) = matrices(&xs, &ys, &ws, 1.0, 1.0);

        let matrix = win_matrix + &tie_matrix / 2.0;

        let expected = array![
            0.050799672530389396,
            0.1311506914535368,
            0.13168568070110523,
            0.33688868041573855,
            0.34947527489923,
        ];

        let (actual, iterations) = bradley_terry(&matrix.view(), tolerance, 100);

        assert_eq!(actual.len(), matrix.shape()[0]);
        assert_ne!(iterations, 0);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = tolerance * 1e1);
        }
    }

    #[test]
    fn test_newman() {
        let tolerance = 1e-8;

        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let ws = ArrayView1::from(&utils::fixtures::WS);

        let (win_matrix, tie_matrix) = matrices(&xs, &ys, &ws, 1.0, 1.0);

        let expected_v = 3.4609664512240546;
        let v_init = 0.5;

        let expected = array![
            0.01797396524323986,
            0.48024292914742794,
            0.4814219179584765,
            12.805581189559241,
            13.036548812823787,
        ];

        let (actual, v, iterations) = newman(
            &win_matrix.view(),
            &tie_matrix.view(),
            v_init,
            tolerance,
            100,
        );

        assert_eq!(actual.len(), win_matrix.shape()[0]);
        assert_eq!(actual.len(), tie_matrix.shape()[0]);
        assert_ne!(iterations, 0);

        assert_ne!(v, v_init);
        assert!(v.is_normal());
        assert_abs_diff_eq!(v, expected_v, epsilon = tolerance * 1e3);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = tolerance * 1e1);
        }
    }
}
