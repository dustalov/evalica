use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::HashMap;

/// Function type for custom distance metrics.
/// Takes a slice of unique values and returns a distance matrix.
pub type DistanceFunc = fn(&[f64]) -> Array2<f64>;
const MIN_RATERS: usize = 2;

/// Distance metric type for Krippendorff's alpha.
#[derive(Clone)]
pub enum Distance {
    Nominal,
    Ordinal,
    Interval,
    Ratio,
    CustomFunc(DistanceFunc),
    CustomMatrix(Array2<f64>),
}

impl Distance {
    /// Parse distance from string.
    ///
    /// # Errors
    ///
    /// Returns an error if `s` does not match a supported distance type.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "nominal" => Ok(Self::Nominal),
            "ordinal" => Ok(Self::Ordinal),
            "interval" => Ok(Self::Interval),
            "ratio" => Ok(Self::Ratio),
            _ => Err(format!("Unknown distance '{s}'")),
        }
    }
}

/// Factorize values in the data matrix, mapping unique values to integer codes.
///
/// # Arguments
///
/// * `data` - 2D array where NaN represents missing values
///
/// # Returns
///
/// A tuple of (`coded_matrix`, `unique_values`) where missing values are coded as `-1`.
fn factorize_values(data: &ArrayView2<f64>) -> (Array2<i64>, Vec<f64>) {
    let mut value_to_code: HashMap<u64, usize> = HashMap::new();
    let mut unique_values: Vec<f64> = Vec::new();
    let mut coded = Array2::from_elem(data.dim(), -1i64);

    for &val in data.iter() {
        if !val.is_nan() {
            let bits = val.to_bits();
            value_to_code.entry(bits).or_insert_with(|| {
                let code = unique_values.len();
                unique_values.push(val);
                code
            });
        }
    }

    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut remap: HashMap<u64, i64> = HashMap::new();
    for (new_idx, &val) in unique_values.iter().enumerate() {
        remap.insert(
            val.to_bits(),
            i64::try_from(new_idx).expect("index should fit into i64"),
        );
    }

    for ((i, j), &val) in data.indexed_iter() {
        if !val.is_nan() {
            coded[(i, j)] = remap[&val.to_bits()];
        }
    }

    (coded, unique_values)
}

/// Build the coincidence matrix from coded unit values.
///
/// # Arguments
///
/// * `matrix_indices` - Coded matrix with -1 for missing values
/// * `n_unique` - Number of unique non-missing values
///
/// # Returns
///
/// The coincidence matrix.
fn coincidence_matrix(matrix_indices: &ArrayView2<i64>, n_unique: usize) -> Array2<f64> {
    let n_units = matrix_indices.nrows();
    let mut c = Array2::<f64>::zeros((n_units, n_unique));

    for (i, unit_row) in matrix_indices.rows().into_iter().enumerate() {
        for &val in unit_row {
            if val >= 0 {
                if let Ok(val_idx) = usize::try_from(val) {
                    c[(i, val_idx)] += 1.0;
                }
            }
        }
    }

    let m_u = c.sum_axis(Axis(1));
    let weights = m_u.mapv(|m| {
        if m >= MIN_RATERS as f64 {
            1.0 / (m - 1.0)
        } else {
            0.0
        }
    });

    let weights_col = weights.insert_axis(Axis(1));
    let cw = &c * &weights_col;
    let mut coincidence = cw.t().dot(&c);

    let diag_adj = cw.sum_axis(Axis(0));
    for i in 0..n_unique {
        coincidence[[i, i]] -= diag_adj[i];
    }

    coincidence
}

/// Compute nominal distance matrix.
fn nominal_distance(n_unique: usize) -> Array2<f64> {
    let mut delta = Array2::<f64>::ones((n_unique, n_unique));
    for i in 0..n_unique {
        delta[[i, i]] = 0.0;
    }
    delta
}

/// Compute ordinal distance matrix based on cumulative frequencies.
fn ordinal_distance(coincidence: &ArrayView2<f64>) -> Array2<f64> {
    let n_c = coincidence.sum_axis(Axis(1));
    let mut cum_freqs = n_c.clone();
    let mut cumsum = 0.0;
    for x in &mut cum_freqs {
        cumsum += *x;
        *x = cumsum;
    }

    let midpoints = &cum_freqs - &(&n_c / 2.0);
    let midpoints_col = midpoints.view().insert_axis(Axis(1));
    let midpoints_row = midpoints.view().insert_axis(Axis(0));
    (&midpoints_col - &midpoints_row).mapv(|x| x.powi(2))
}

/// Compute interval distance matrix.
fn interval_distance(unique_values: &[f64]) -> Array2<f64> {
    let values = Array1::from_vec(unique_values.to_vec());
    let v_col = values.view().insert_axis(Axis(1));
    let v_row = values.view().insert_axis(Axis(0));
    (&v_col - &v_row).mapv(|x| x.powi(2))
}

/// Compute ratio distance matrix.
fn ratio_distance(unique_values: &[f64]) -> Array2<f64> {
    let values = Array1::from_vec(unique_values.to_vec());
    let v_col = values.view().insert_axis(Axis(1));
    let v_row = values.view().insert_axis(Axis(0));
    let sum_matrix = &v_col + &v_row;
    let diff_matrix = &v_col - &v_row;

    let mut delta = Array2::<f64>::zeros(sum_matrix.dim());
    for ((i, j), &s) in sum_matrix.indexed_iter() {
        if s != 0.0 {
            delta[(i, j)] = (diff_matrix[(i, j)] / s).powi(2);
        }
    }
    delta
}

/// Compute the expected disagreement matrix.
fn compute_expected_matrix(coincidence: &ArrayView2<f64>) -> Array2<f64> {
    let n_c = coincidence.sum_axis(Axis(1));
    let n_total = n_c.sum();

    if n_total > 1.0 {
        let n_c_col = n_c.view().insert_axis(Axis(1));
        let n_c_row = n_c.view().insert_axis(Axis(0));
        let mut expected = &n_c_col * &n_c_row;
        for i in 0..n_c.len() {
            expected[[i, i]] = n_c[i] * (n_c[i] - 1.0);
        }
        expected / (n_total - 1.0)
    } else {
        Array2::<f64>::zeros(coincidence.dim())
    }
}

fn delta_matrix_from_distance(
    distance: Distance,
    n_unique: usize,
    unique_values: &[f64],
    coincidence: &ArrayView2<f64>,
) -> Array2<f64> {
    match distance {
        Distance::Nominal => nominal_distance(n_unique),
        Distance::Ordinal => ordinal_distance(coincidence),
        Distance::Interval => interval_distance(unique_values),
        Distance::Ratio => ratio_distance(unique_values),
        Distance::CustomFunc(func) => func(unique_values),
        Distance::CustomMatrix(matrix) => matrix,
    }
}

struct AlphaComputed {
    delta: Array2<f64>,
    observed_disagreement: f64,
    expected_disagreement: f64,
}

pub struct AlphaBootstrap {
    pub alpha: f64,
    pub observed: f64,
    pub expected: f64,
    pub distribution: Vec<f64>,
}

fn alpha_compute_from_factorized(
    matrix_indices: &ArrayView2<i64>,
    unique_values: &[f64],
    distance: Distance,
) -> AlphaComputed {
    let n_unique = unique_values.len();
    let coincidence = coincidence_matrix(matrix_indices, n_unique);
    let expected = compute_expected_matrix(&coincidence.view());
    let delta = delta_matrix_from_distance(distance, n_unique, unique_values, &coincidence.view());
    let observed_disagreement: f64 = (&coincidence * &delta).sum();
    let expected_disagreement: f64 = (&expected * &delta).sum();

    AlphaComputed {
        delta,
        observed_disagreement,
        expected_disagreement,
    }
}

fn unit_counts(matrix_indices: &ArrayView2<i64>) -> Vec<usize> {
    matrix_indices
        .rows()
        .into_iter()
        .map(|row| row.iter().filter(|&&v| v >= 0).count())
        .collect()
}

fn compute_pair_errors(
    matrix_indices: &ArrayView2<i64>,
    delta: &ArrayView2<f64>,
    expected_disagreement: f64,
) -> Vec<f64> {
    let mut errors = Vec::new();

    for row in matrix_indices.rows() {
        let valid: Vec<usize> = row
            .iter()
            .filter_map(|&v| usize::try_from(v).ok())
            .collect();

        if valid.len() < 2 {
            continue;
        }

        for i in 0..(valid.len() - 1) {
            for j in (i + 1)..valid.len() {
                let d = delta[[valid[i], valid[j]]];
                errors.push((2.0 * d) / expected_disagreement);
            }
        }
    }

    errors
}

/// Compute Krippendorff's alpha from factorized data.
///
/// # Arguments
///
/// * `matrix_indices` - Coded matrix with -1 for missing values
/// * `unique_values` - Unique values corresponding to the codes
/// * `distance` - Distance metric to use
///
/// # Returns
///
/// A tuple of (`alpha`, `observed_disagreement`, `expected_disagreement`).
///
/// # Errors
///
/// Returns an error when the provided distance value cannot be processed.
pub fn alpha_from_factorized(
    matrix_indices: &ArrayView2<i64>,
    unique_values: &[f64],
    distance: Distance,
) -> Result<(f64, f64, f64), String> {
    let computed = alpha_compute_from_factorized(matrix_indices, unique_values, distance);

    let alpha_value = if computed.expected_disagreement == 0.0 {
        0.0
    } else {
        1.0 - computed.observed_disagreement / computed.expected_disagreement
    };

    Ok((
        alpha_value,
        computed.observed_disagreement,
        computed.expected_disagreement,
    ))
}

/// Compute Krippendorff's alpha confidence intervals with KALPHA-style bootstrap.
///
/// # Arguments
///
/// * `matrix_indices` - Coded matrix with -1 for missing values
/// * `unique_values` - Unique values corresponding to the codes
/// * `distance` - Distance metric to use
/// * `n_resamples` - Number of bootstrap samples; truncated to nearest lower multiple of `min_resamples`
/// * `min_resamples` - Minimum bootstrap sample count and truncation step
/// * `seed` - Optional RNG seed
///
/// # Errors
///
/// Returns an error when the bootstrap count is invalid or expected disagreement is zero.
pub fn alpha_bootstrap_from_factorized(
    matrix_indices: &ArrayView2<i64>,
    unique_values: &[f64],
    distance: Distance,
    n_resamples: usize,
    min_resamples: usize,
    seed: Option<u64>,
) -> Result<AlphaBootstrap, String> {
    if min_resamples == 0 {
        return Err("min_resamples must be a positive integer.".to_string());
    }

    if n_resamples < min_resamples {
        return Err(format!(
            "Number of resamples must be at least {min_resamples}."
        ));
    }

    let computed = alpha_compute_from_factorized(matrix_indices, unique_values, distance);
    if computed.expected_disagreement <= 0.0 {
        return Err("Bootstrapping is not defined when expected disagreement is zero.".to_string());
    }

    let unit_rater_counts = unit_counts(matrix_indices);
    let pair_errors_vec = compute_pair_errors(
        matrix_indices,
        &computed.delta.view(),
        computed.expected_disagreement,
    );
    if pair_errors_vec.is_empty() {
        return Err("Bootstrapping cannot proceed without pairable units.".to_string());
    }
    let pair_errors = Array1::from(pair_errors_vec);

    let mut distribution = Array1::from_elem(n_resamples, 1.0);
    let mut rng: StdRng =
        seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);

    let mut previous_draws = Array1::<usize>::zeros(n_resamples);

    for &raters_per_unit in &unit_rater_counts {
        if raters_per_unit < MIN_RATERS {
            continue;
        }

        let n_draws = (raters_per_unit * (raters_per_unit - 1)) / 2;
        let weight = 1.0 / (raters_per_unit - 1) as f64;

        for draw_index in 1..=n_draws {
            let current_draws =
                Array1::from_shape_fn(n_resamples, |_| rng.random_range(0..pair_errors.len()));

            if draw_index == 2 {
                let mut current_draws = current_draws;
                for r in 0..n_resamples {
                    if current_draws[r] == previous_draws[r] {
                        current_draws[r] = rng.random_range(0..pair_errors.len());
                    }
                }
                for r in 0..n_resamples {
                    distribution[r] -= pair_errors[current_draws[r]] * weight;
                }
            } else {
                if draw_index == 1 {
                    previous_draws.assign(&current_draws);
                }
                for r in 0..n_resamples {
                    distribution[r] -= pair_errors[current_draws[r]] * weight;
                }
            }
        }
    }

    distribution.mapv_inplace(|a: f64| if a < -1.0 { -1.0 } else { a });

    let alpha = if computed.expected_disagreement == 0.0 {
        0.0
    } else {
        1.0 - computed.observed_disagreement / computed.expected_disagreement
    };

    Ok(AlphaBootstrap {
        alpha,
        observed: computed.observed_disagreement,
        expected: computed.expected_disagreement,
        distribution: distribution.to_vec(),
    })
}

/// Compute Krippendorff's alpha.
///
/// # Arguments
///
/// * `data` - 2D array with units as rows and raters as columns; NaN for missing values
/// * `distance` - Distance metric to use
///
/// # Returns
///
/// A tuple of (`alpha`, `observed_disagreement`, `expected_disagreement`).
///
/// # Errors
///
/// Returns an error if no units have at least 2 ratings or if the distance is unknown.
pub fn alpha(data: &ArrayView2<f64>, distance: Distance) -> Result<(f64, f64, f64), String> {
    let mut has_valid_unit = false;
    for row in data.rows() {
        let valid_count = row.iter().filter(|&&v| !v.is_nan()).count();
        if valid_count >= 2 {
            has_valid_unit = true;
            break;
        }
    }

    if !has_valid_unit {
        return Err("No units have at least 2 ratings.".to_string());
    }

    let (matrix_indices, unique_values) = factorize_values(data);

    alpha_from_factorized(&matrix_indices.view(), &unique_values, distance)
}

/// Example custom distance function: squared difference.
/// This is equivalent to the interval distance metric.
#[must_use]
pub fn custom_squared_diff(unique_values: &[f64]) -> Array2<f64> {
    let values = Array1::from_vec(unique_values.to_vec());
    let v_col = values.view().insert_axis(Axis(1));
    let v_row = values.view().insert_axis(Axis(0));
    (&v_col - &v_row).mapv(|x| x.powi(2))
}

/// Example custom distance function: absolute difference.
#[must_use]
pub fn custom_abs_diff(unique_values: &[f64]) -> Array2<f64> {
    let values = Array1::from_vec(unique_values.to_vec());
    let v_col = values.view().insert_axis(Axis(1));
    let v_row = values.view().insert_axis(Axis(0));
    (&v_col - &v_row).mapv(f64::abs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq as assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_factorize_values() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [f64::NAN, 1.0]];
        let (coded, unique) = factorize_values(&data.view());

        assert_eq!(unique, vec![1.0, 2.0, 3.0]);
        assert_eq!(coded[[0, 0]], 0);
        assert_eq!(coded[[0, 1]], 1);
        assert_eq!(coded[[1, 0]], 1);
        assert_eq!(coded[[1, 1]], 2);
        assert_eq!(coded[[2, 0]], -1);
        assert_eq!(coded[[2, 1]], 0);
    }

    #[test]
    fn test_nominal_distance() {
        let delta = nominal_distance(3);
        assert_approx_eq!(delta[[0, 0]], 0.0, epsilon = f64::EPSILON);
        assert_approx_eq!(delta[[0, 1]], 1.0, epsilon = f64::EPSILON);
        assert_approx_eq!(delta[[1, 2]], 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_custom_distance_squared_diff() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result_interval = alpha(&data.view(), Distance::Interval).unwrap();
        let result_custom = alpha(&data.view(), Distance::CustomFunc(custom_squared_diff)).unwrap();

        assert_approx_eq!(result_interval.0, result_custom.0, epsilon = 1e-10);
        assert_approx_eq!(result_interval.1, result_custom.1, epsilon = 1e-10);
        assert_approx_eq!(result_interval.2, result_custom.2, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_distance_abs_diff() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = alpha(&data.view(), Distance::CustomFunc(custom_abs_diff)).unwrap();

        assert!(result.0.is_finite());
        assert!(result.1 >= 0.0);
        assert!(result.2 >= 0.0);
    }

    #[test]
    fn test_custom_distance_with_missing_values() {
        let data = array![[1.0, f64::NAN], [2.0, 3.0], [3.0, 4.0]];

        let result = alpha(&data.view(), Distance::CustomFunc(custom_squared_diff)).unwrap();

        assert!(result.0.is_finite());
        assert!(result.1 >= 0.0);
        assert!(result.2 >= 0.0);
    }

    #[test]
    fn test_alpha_bootstrap_nominal_reference() {
        let data = array![
            [1.0, 1.0, f64::NAN, 1.0],
            [2.0, 2.0, 3.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 4.0, 4.0, 4.0],
            [1.0, 1.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [f64::NAN, 5.0, 5.0, 5.0],
            [f64::NAN, f64::NAN, 1.0, 1.0]
        ];

        let (codes, unique_values) = factorize_values(&data.view());
        let result = alpha_bootstrap_from_factorized(
            &codes.view(),
            &unique_values,
            Distance::Nominal,
            5000,
            1000,
            Some(12345),
        )
        .unwrap();

        assert_approx_eq!(result.alpha, 0.7434211, epsilon = 1e-6);
        assert_eq!(result.distribution.len(), 5000);
    }

    #[test]
    fn test_alpha_bootstrap_invalid_bootstrap_count() {
        let data = array![[1.0, 2.0], [2.0, 3.0]];
        let (codes, unique_values) = factorize_values(&data.view());
        let result = alpha_bootstrap_from_factorized(
            &codes.view(),
            &unique_values,
            Distance::Nominal,
            999,
            1000,
            Some(1),
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_alpha_bootstrap_custom_minimum_bootstrap_count() {
        let data = array![[1.0, 2.0], [2.0, 3.0]];
        let (codes, unique_values) = factorize_values(&data.view());
        let result = alpha_bootstrap_from_factorized(
            &codes.view(),
            &unique_values,
            Distance::Nominal,
            1499,
            1500,
            Some(1),
        );

        assert!(result.is_err());
        assert!(result
            .err()
            .expect("error expected")
            .contains("at least 1500"));
    }

    #[test]
    fn test_alpha_bootstrap_invalid_min_resamples() {
        let data = array![[1.0, 2.0], [2.0, 3.0]];
        let (codes, unique_values) = factorize_values(&data.view());
        let result = alpha_bootstrap_from_factorized(
            &codes.view(),
            &unique_values,
            Distance::Nominal,
            1000,
            0,
            Some(1),
        );

        assert!(result.is_err());
        assert!(result
            .err()
            .expect("error expected")
            .contains("min_resamples must be a positive integer"));
    }
}
