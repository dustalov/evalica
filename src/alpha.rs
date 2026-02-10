use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;

/// Function type for custom distance metrics.
/// Takes a slice of unique values and returns a distance matrix.
pub type DistanceFunc = fn(&[f64]) -> Array2<f64>;

/// Distance metric type for Krippendorff's alpha.
#[derive(Clone)]
pub enum Distance {
    Nominal,
    Ordinal,
    Interval,
    Ratio,
    Custom(DistanceFunc),
}

impl Distance {
    /// Parse distance from string.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "nominal" => Ok(Self::Nominal),
            "ordinal" => Ok(Self::Ordinal),
            "interval" => Ok(Self::Interval),
            "ratio" => Ok(Self::Ratio),
            _ => Err(format!("Unknown distance '{}'", s)),
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
/// A tuple of (coded_matrix, unique_values) where missing values are coded as -1.
fn factorize_values(data: &ArrayView2<f64>) -> (Array2<i64>, Vec<f64>) {
    let mut value_to_code: HashMap<u64, usize> = HashMap::new();
    let mut unique_values: Vec<f64> = Vec::new();
    let mut coded = Array2::from_elem(data.dim(), -1i64);

    for &val in data.iter() {
        if !val.is_nan() {
            let bits = val.to_bits();
            if !value_to_code.contains_key(&bits) {
                value_to_code.insert(bits, unique_values.len());
                unique_values.push(val);
            }
        }
    }

    unique_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut remap: HashMap<u64, i64> = HashMap::new();
    for (new_idx, &val) in unique_values.iter().enumerate() {
        remap.insert(val.to_bits(), new_idx as i64);
    }

    for ((i, j), &val) in data.indexed_iter() {
        if !val.is_nan() {
            coded[[i, j]] = remap[&val.to_bits()];
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
    let min_raters = 2;

    for unit_row in matrix_indices.rows() {
        let valid_values: Vec<i64> = unit_row.iter().filter(|&&v| v >= 0).copied().collect();
        let m_u = valid_values.len();

        if m_u < min_raters {
            continue;
        }

        let weight = 1.0 / (m_u - 1) as f64;

        let mut counts = vec![0.0; n_unique];
        for &val in &valid_values {
            counts[val as usize] += 1.0;
        }

        for i in 0..n_unique {
            for j in 0..n_unique {
                let contribution = if i == j {
                    counts[i] * (counts[i] - 1.0)
                } else {
                    counts[i] * counts[j]
                };
                coincidence[[i, j]] += weight * contribution;
            }
        }
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
    let n_c: Vec<f64> = (0..coincidence.nrows())
        .map(|i| coincidence.row(i).sum())
        .collect();

    let mut cum_freqs = Vec::with_capacity(n_c.len());
    let mut cumsum = 0.0;
    for &freq in &n_c {
        cumsum += freq;
        cum_freqs.push(cumsum);
    }

    let midpoints: Vec<f64> = cum_freqs
        .iter()
        .zip(&n_c)
        .map(|(&cum, &freq)| cum - freq / 2.0)
        .collect();

    let n = midpoints.len();
    let mut delta = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            delta[[i, j]] = (midpoints[i] - midpoints[j]).powi(2);
        }
    }

    delta
}

/// Compute interval distance matrix.
fn interval_distance(unique_values: &[f64]) -> Array2<f64> {
    let n = unique_values.len();
    let mut delta = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            delta[[i, j]] = (unique_values[i] - unique_values[j]).powi(2);
        }
    }
    delta
}

/// Compute ratio distance matrix.
fn ratio_distance(unique_values: &[f64]) -> Array2<f64> {
    let n = unique_values.len();
    let mut delta = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let sum = unique_values[i] + unique_values[j];
            let diff = unique_values[i] - unique_values[j];
            if sum != 0.0 {
                delta[[i, j]] = (diff / sum).powi(2);
            }
        }
    }
    delta
}

/// Compute the expected disagreement matrix.
fn compute_expected_matrix(coincidence: &ArrayView2<f64>) -> Array2<f64> {
    let n_c: Vec<f64> = (0..coincidence.nrows())
        .map(|i| coincidence.row(i).sum())
        .collect();
    let n_total: f64 = n_c.iter().sum();

    if n_total > 1.0 {
        let n = n_c.len();
        let mut expected = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                expected[[i, j]] = if i == j {
                    n_c[i] * (n_c[i] - 1.0)
                } else {
                    n_c[i] * n_c[j]
                };
            }
        }
        expected / (n_total - 1.0)
    } else {
        Array2::<f64>::zeros(coincidence.dim())
    }
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
/// A tuple of (alpha, observed_disagreement, expected_disagreement).
pub fn alpha_from_factorized(
    matrix_indices: &ArrayView2<i64>,
    unique_values: &[f64],
    distance: Distance,
) -> Result<(f64, f64, f64), String> {
    let n_unique = unique_values.len();
    let coincidence = coincidence_matrix(matrix_indices, n_unique);
    let expected = compute_expected_matrix(&coincidence.view());

    let delta = match distance {
        Distance::Nominal => nominal_distance(n_unique),
        Distance::Ordinal => ordinal_distance(&coincidence.view()),
        Distance::Interval => interval_distance(unique_values),
        Distance::Ratio => ratio_distance(unique_values),
        Distance::Custom(func) => func(unique_values),
    };

    let observed_disagreement: f64 = (&coincidence * &delta).sum();
    let expected_disagreement: f64 = (&expected * &delta).sum();

    let alpha_value = if expected_disagreement == 0.0 {
        0.0
    } else {
        1.0 - observed_disagreement / expected_disagreement
    };

    Ok((alpha_value, observed_disagreement, expected_disagreement))
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
/// A tuple of (alpha, observed_disagreement, expected_disagreement).
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
pub fn custom_squared_diff(unique_values: &[f64]) -> Array2<f64> {
    let n = unique_values.len();
    let mut delta = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let diff = unique_values[i] - unique_values[j];
            delta[[i, j]] = diff * diff;
        }
    }
    delta
}

/// Example custom distance function: absolute difference.
pub fn custom_abs_diff(unique_values: &[f64]) -> Array2<f64> {
    let n = unique_values.len();
    let mut delta = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let diff = (unique_values[i] - unique_values[j]).abs();
            delta[[i, j]] = diff;
        }
    }
    delta
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(delta[[0, 0]], 0.0);
        assert_eq!(delta[[0, 1]], 1.0);
        assert_eq!(delta[[1, 2]], 1.0);
    }

    #[test]
    fn test_custom_distance_squared_diff() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result_interval = alpha(&data.view(), Distance::Interval).unwrap();
        let result_custom = alpha(&data.view(), Distance::Custom(custom_squared_diff)).unwrap();

        assert!((result_interval.0 - result_custom.0).abs() < 1e-10);
        assert!((result_interval.1 - result_custom.1).abs() < 1e-10);
        assert!((result_interval.2 - result_custom.2).abs() < 1e-10);
    }

    #[test]
    fn test_custom_distance_abs_diff() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let result = alpha(&data.view(), Distance::Custom(custom_abs_diff)).unwrap();

        assert!(result.0.is_finite());
        assert!(result.1 >= 0.0);
        assert!(result.2 >= 0.0);
    }

    #[test]
    fn test_custom_distance_with_missing_values() {
        let data = array![[1.0, f64::NAN], [2.0, 3.0], [3.0, 4.0]];

        let result = alpha(&data.view(), Distance::Custom(custom_squared_diff)).unwrap();

        assert!(result.0.is_finite());
        assert!(result.1 >= 0.0);
        assert!(result.2 >= 0.0);
    }
}
