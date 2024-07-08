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

        scores_new.fill(0.0);

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
    assert!(v_init.is_normal());
    assert!(v_init > 0.0);

    let win_tie_half = win_matrix + &(tie_matrix * 0.5);
    let mut scores = Array1::ones(win_matrix.shape()[0]);
    let mut v = v_init;
    let mut iterations = 0;

    loop {
        iterations += 1;

        let scores_broadcast = scores.broadcast((scores.len(), scores.len())).unwrap();
        let scores_outer_sqrt =
            (scores_broadcast.to_owned() * scores_broadcast.t()).mapv_into(f64::sqrt);

        let term = &scores_broadcast + &scores_broadcast.t() + &(2.0 * v * &scores_outer_sqrt);

        let v_numerator = tie_matrix * &(&scores_broadcast + &scores_broadcast.t()) / &term;
        let v_numerator_sum = v_numerator.sum() / 2.0;

        let v_denominator = win_matrix * &(2.0 * &scores_outer_sqrt) / &term;
        let v_denominator_sum = v_denominator.sum();

        v = v_numerator_sum / v_denominator_sum;

        if v.is_nan() {
            v = tolerance;
        }

        let scores_old = scores.clone();

        let pi_numerator = &win_tie_half * &(&scores_broadcast + &(v * &scores_outer_sqrt)) / &term;
        let pi_numerator_sum = pi_numerator.sum_axis(Axis(1));

        let pi_denominator = &win_tie_half * &(1.0 + v * &scores_outer_sqrt) / &term;
        let pi_denominator_sum = pi_denominator.sum_axis(Axis(0));

        scores = &pi_numerator_sum / &pi_denominator_sum;

        scores.iter_mut().for_each(|x| {
            if x.is_nan() {
                *x = tolerance
            }
        });

        let scores_normalized = &scores / (&scores + 1.0);
        let scores_old_normalized = &scores_old / (&scores_old + 1.0);

        let converged = scores_normalized
            .iter()
            .zip(scores_old_normalized.iter())
            .all(|(a, b)| (a - b).abs() <= tolerance * f64::max(a.abs(), b.abs()) + tolerance)
            || iterations >= limit;

        if converged {
            break;
        }
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

        println!("{:?}", actual);

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

        let expected_v = 1.483370503346757;
        let v_init = 0.5;

        let expected = array![
            0.29236875388822886,
            0.8129264752163917,
            1.6686265317109035,
            1.919540868932138,
            0.8230632211243398,
        ];

        let (actual, v, iterations) = newman(
            &win_matrix.view(),
            &tie_matrix.view(),
            v_init,
            tolerance,
            100,
        );

        println!("{:?}", win_matrix);
        println!();
        println!("{:?}", tie_matrix);
        println!();
        println!("{:?}", actual);

        assert_eq!(actual.len(), win_matrix.shape()[0]);
        assert_eq!(actual.len(), tie_matrix.shape()[0]);
        assert_ne!(iterations, 0);

        assert_ne!(v, v_init);
        assert_abs_diff_eq!(v, expected_v, epsilon = tolerance * 1e1);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = tolerance * 1e1);
        }
    }
}
