//! Implements Krippendorff's alpha for inter-rater reliability.
//!
//! Krippendorff's alpha is a statistical measure of the agreement achieved when coding a set
//! of units. Unlike many other reliability coefficients, it generalizes across different
//! measurement scales (nominal, ordinal, interval, ratio) and can handle missing data.
//!
//! # References
//!
//! - Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology*.
//!   Sage Publications, Thousand Oaks, CA.
//! - Krippendorff, K. (2006). [*Bootstrapping Distributions for Krippendorff’s Alpha*][boot-algo].
//! - Hayes, A. F. [*Statistical Methods and Macros for SPSS, SAS, and R*][hayes-macros].
//!
//! [boot-algo]: https://www.asc.upenn.edu/sites/default/files/2021-03/Algorithm%20for%20Bootstrapping%20a%20Distribution%20of%20Alpha.pdf
//! [hayes-macros]: https://afhayes.com/spss-sas-and-r-macros-and-code.html

use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Zip};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::{HashMap, HashSet};

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
    let mut seen: HashSet<u64> = HashSet::new();
    let mut unique_values: Vec<f64> = Vec::new();
    let mut coded = Array2::from_elem(data.dim(), -1i64);

    for &val in data.iter().filter(|v| !v.is_nan()) {
        if seen.insert(val.to_bits()) {
            unique_values.push(val);
        }
    }

    unique_values.sort_by(f64::total_cmp);

    let remap: HashMap<u64, i64> = unique_values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            (
                v.to_bits(),
                i64::try_from(i).expect("index should fit into i64"),
            )
        })
        .collect();

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
    let mut coincidence = Array2::<f64>::zeros((n_unique, n_unique));
    let mut counts = vec![0.0; n_unique];
    let mut present = Vec::with_capacity(n_unique);

    for unit_row in matrix_indices.rows() {
        for &val in unit_row {
            if let Ok(val_idx) = usize::try_from(val) {
                if counts[val_idx] == 0.0 {
                    present.push(val_idx);
                }
                counts[val_idx] += 1.0;
            }
        }

        let m_u: f64 = present.iter().map(|&v| counts[v]).sum();

        if m_u >= MIN_RATERS as f64 {
            let weight = 1.0 / (m_u - 1.0);

            for &a in &present {
                let weighted = weight * counts[a];

                for &b in &present {
                    coincidence[[a, b]] = weighted.mul_add(counts[b], coincidence[[a, b]]);
                }

                coincidence[[a, a]] -= weighted;
            }
        }

        for &v in &present {
            counts[v] = 0.0;
        }
        present.clear();
    }

    coincidence
}

/// Compute nominal distance matrix.
fn nominal_distance(n_unique: usize) -> Array2<f64> {
    Array2::from_shape_fn(
        (n_unique, n_unique),
        |(i, j)| if i == j { 0.0 } else { 1.0 },
    )
}

/// Compute ordinal distance matrix based on cumulative frequencies.
fn ordinal_distance(marginals: &ArrayView1<f64>) -> Array2<f64> {
    let mut cumsum = 0.0;
    let midpoints: Vec<f64> = marginals
        .iter()
        .map(|&n| {
            cumsum += n;
            cumsum - n / 2.0
        })
        .collect();

    let n_unique = midpoints.len();
    Array2::from_shape_fn((n_unique, n_unique), |(i, j)| {
        (midpoints[i] - midpoints[j]).powi(2)
    })
}

/// Compute interval distance matrix.
fn interval_distance(unique_values: &[f64]) -> Array2<f64> {
    let n_unique = unique_values.len();
    Array2::from_shape_fn((n_unique, n_unique), |(i, j)| {
        (unique_values[i] - unique_values[j]).powi(2)
    })
}

/// Compute ratio distance matrix.
fn ratio_distance(unique_values: &[f64]) -> Array2<f64> {
    let n_unique = unique_values.len();
    Array2::from_shape_fn((n_unique, n_unique), |(i, j)| {
        let sum = unique_values[i] + unique_values[j];
        if sum == 0.0 {
            0.0
        } else {
            ((unique_values[i] - unique_values[j]) / sum).powi(2)
        }
    })
}

/// Compute the expected disagreement from the coincidence marginals.
fn compute_expected_disagreement(marginals: &ArrayView1<f64>, delta: &ArrayView2<f64>) -> f64 {
    let n_total = marginals.sum();

    if n_total <= 1.0 {
        return 0.0;
    }

    Zip::indexed(delta).fold(0.0, |acc, (i, j), &d| {
        let expected = if i == j {
            marginals[i] * (marginals[i] - 1.0)
        } else {
            marginals[i] * marginals[j]
        };
        (expected / (n_total - 1.0)).mul_add(d, acc)
    })
}

fn delta_matrix_from_distance(
    distance: Distance,
    n_unique: usize,
    unique_values: &[f64],
    marginals: &ArrayView1<f64>,
) -> Array2<f64> {
    match distance {
        Distance::Nominal => nominal_distance(n_unique),
        Distance::Ordinal => ordinal_distance(marginals),
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
    let marginals = coincidence.sum_axis(Axis(1));
    let delta = delta_matrix_from_distance(distance, n_unique, unique_values, &marginals.view());
    let observed_disagreement = Zip::from(&coincidence)
        .and(&delta)
        .fold(0.0, |acc, &c, &d| c.mul_add(d, acc));
    let expected_disagreement = compute_expected_disagreement(&marginals.view(), &delta.view());

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
    let mut valid = Vec::with_capacity(matrix_indices.ncols());

    for row in matrix_indices.rows() {
        valid.clear();
        valid.extend(row.iter().filter_map(|&v| usize::try_from(v).ok()));

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
/// This function calculates the inter-rater reliability coefficient &alpha; based on
/// coincidence and expected disagreement matrices, as described by Krippendorff (2018).
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
/// This method implements the bootstrapping algorithm described by Krippendorff (2006)
/// and Hayes & Krippendorff (2007) to estimate the reliability of the alpha coefficient.
///
/// # Arguments
///
/// * `matrix_indices` - Coded matrix with -1 for missing values
/// * `unique_values` - Unique values corresponding to the codes
/// * `distance` - Distance metric to use
/// * `n_resamples` - Number of bootstrap samples
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
    seed: Option<u64>,
) -> Result<AlphaBootstrap, String> {
    if n_resamples == 0 {
        return Err("Number of resamples must be a positive integer.".to_string());
    }

    let computed = alpha_compute_from_factorized(matrix_indices, unique_values, distance);
    if computed.expected_disagreement <= 0.0 {
        return Err("Bootstrapping is not defined when expected disagreement is zero.".to_string());
    }

    let unit_rater_counts = unit_counts(matrix_indices);
    let pair_errors = compute_pair_errors(
        matrix_indices,
        &computed.delta.view(),
        computed.expected_disagreement,
    );
    if pair_errors.is_empty() {
        return Err("Bootstrapping cannot proceed without pairable units.".to_string());
    }

    let mut distribution = vec![1.0; n_resamples];
    let mut rng: StdRng =
        seed.map_or_else(|| StdRng::from_rng(&mut rand::rng()), StdRng::seed_from_u64);

    let mut previous_draws = vec![0; n_resamples];
    let mut current_draws = vec![0; n_resamples];

    for &raters_per_unit in &unit_rater_counts {
        if raters_per_unit < MIN_RATERS {
            continue;
        }

        let n_draws = (raters_per_unit * (raters_per_unit - 1)) / 2;
        let weight = 1.0 / (raters_per_unit - 1) as f64;

        for draw_index in 1..=n_draws {
            for draw in &mut current_draws {
                *draw = rng.random_range(0..pair_errors.len());
            }

            match draw_index {
                1 => previous_draws.copy_from_slice(&current_draws),
                2 => {
                    for (current, &previous) in current_draws.iter_mut().zip(&previous_draws) {
                        if *current == previous {
                            *current = rng.random_range(0..pair_errors.len());
                        }
                    }
                }
                _ => {}
            }

            for (value, &draw) in distribution.iter_mut().zip(&current_draws) {
                *value = pair_errors[draw].mul_add(-weight, *value);
            }
        }
    }

    for value in &mut distribution {
        *value = value.max(-1.0);
    }

    let alpha = if computed.expected_disagreement == 0.0 {
        0.0
    } else {
        1.0 - computed.observed_disagreement / computed.expected_disagreement
    };

    Ok(AlphaBootstrap {
        alpha,
        observed: computed.observed_disagreement,
        expected: computed.expected_disagreement,
        distribution,
    })
}

/// Compute Krippendorff's alpha.
///
/// This is a statistical measure of inter-rater reliability that generalizes several
/// known reliability coefficients. It can handle missing data and different types of
/// measurements (nominal, ordinal, interval, ratio).
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq as assert_approx_eq;
    use ndarray::array;

    fn abs_diff(unique_values: &[f64]) -> Array2<f64> {
        let n_unique = unique_values.len();
        Array2::from_shape_fn((n_unique, n_unique), |(i, j)| {
            (unique_values[i] - unique_values[j]).abs()
        })
    }

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
        let result_custom = alpha(&data.view(), Distance::CustomFunc(interval_distance)).unwrap();

        assert_approx_eq!(result_interval.0, result_custom.0, epsilon = 1e-10);
        assert_approx_eq!(result_interval.1, result_custom.1, epsilon = 1e-10);
        assert_approx_eq!(result_interval.2, result_custom.2, epsilon = 1e-10);
    }

    #[test]
    fn test_custom_distance_abs_diff() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = alpha(&data.view(), Distance::CustomFunc(abs_diff)).unwrap();

        assert!(result.0.is_finite());
        assert!(result.1 >= 0.0);
        assert!(result.2 >= 0.0);
    }

    #[test]
    fn test_custom_distance_with_missing_values() {
        let data = array![[1.0, f64::NAN], [2.0, 3.0], [3.0, 4.0]];

        let result = alpha(&data.view(), Distance::CustomFunc(interval_distance)).unwrap();

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
            0,
            Some(1),
        );

        assert!(result.is_err());
        assert!(
            result
                .err()
                .expect("error expected")
                .contains("positive integer")
        );
    }
}
