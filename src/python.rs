use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2};
use pyo3::import_exception;
use pyo3::prelude::*;

use crate::alpha;
use crate::bradley_terry::{bradley_terry, newman};
use crate::counting::{average_win_rate, counting};
use crate::elo::elo;
use crate::linalg::{eigen, pagerank};
use crate::utils::{matrices, nan_to_num, pairwise_scores, win_plus_tie_matrix};

import_exception!(evalica, LengthMismatchError);
import_exception!(evalica, InsufficientRatingsError);
import_exception!(evalica, UnknownDistanceError);

#[pyfunction(name = "matrices")]
fn matrices_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    match matrices(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
    ) {
        Ok((wins, ties)) => Ok((
            wins.into_pyarray(py).unbind(),
            ties.into_pyarray(py).unbind(),
        )),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction(name = "pairwise_scores")]
fn pairwise_scores_pyo3<'py>(
    py: Python<'py>,
    scores: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArray2<f64>>> {
    let scores: PyArrayLike1<'py, f64> = scores.extract()?;
    let pairwise = pairwise_scores(&scores.as_array());

    Ok(pairwise.into_pyarray(py).unbind())
}

#[pyfunction(name = "counting")]
fn counting_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    counting(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
        win_weight,
        tie_weight,
    )
    .map_or_else(
        |_| Err(LengthMismatchError::new_err("mismatching input shapes")),
        |scores| Ok(scores.into_pyarray(py).unbind()),
    )
}

#[pyfunction(name = "average_win_rate")]
fn average_win_rate_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    average_win_rate(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
        win_weight,
        tie_weight,
    )
    .map_or_else(
        |_| Err(LengthMismatchError::new_err("mismatching input shapes")),
        |scores| Ok(scores.into_pyarray(py).unbind()),
    )
}

#[pyfunction(name = "bradley_terry")]
fn bradley_terry_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    match matrices(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
    ) {
        Ok((win_matrix, tie_matrix)) => {
            let matrix = win_plus_tie_matrix(
                &win_matrix.view(),
                &tie_matrix.view(),
                win_weight,
                tie_weight,
                tolerance,
            );

            match bradley_terry(&matrix.view(), tolerance, limit) {
                Ok((scores, iterations)) => Ok((scores.into_pyarray(py).unbind(), iterations)),
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction(name = "newman")]
fn newman_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    v_init: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, f64, usize)> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    match matrices(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
    ) {
        Ok((mut win_matrix, mut tie_matrix)) => {
            nan_to_num(&mut win_matrix, tolerance);
            win_matrix *= win_weight;

            nan_to_num(&mut tie_matrix, tolerance);
            tie_matrix *= tie_weight;

            match newman(
                &win_matrix.view(),
                &tie_matrix.view(),
                v_init,
                tolerance,
                limit,
            ) {
                Ok((scores, v, iterations)) => {
                    Ok((scores.into_pyarray(py).unbind(), v, iterations))
                }
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction(name = "elo")]
fn elo_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    initial: f64,
    base: f64,
    scale: f64,
    k: f64,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    elo(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
        initial,
        base,
        scale,
        k,
        win_weight,
        tie_weight,
    )
    .map_or_else(
        |_| Err(LengthMismatchError::new_err("mismatching input shapes")),
        |scores| Ok(scores.into_pyarray(py).unbind()),
    )
}

#[pyfunction(name = "eigen")]
fn eigen_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    match matrices(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
    ) {
        Ok((win_matrix, tie_matrix)) => {
            let matrix = win_plus_tie_matrix(
                &win_matrix.view(),
                &tie_matrix.view(),
                win_weight,
                tie_weight,
                tolerance,
            );

            match eigen(&matrix.view(), tolerance, limit) {
                Ok((scores, iterations)) => Ok((scores.into_pyarray(py).unbind(), iterations)),
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction(name = "pagerank")]
fn pagerank_pyo3<'py>(
    py: Python<'py>,
    xs: &Bound<'py, PyAny>,
    ys: &Bound<'py, PyAny>,
    winners: &Bound<'py, PyAny>,
    weights: &Bound<'py, PyAny>,
    total: usize,
    damping: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let xs: PyArrayLike1<'py, usize> = xs.extract()?;
    let ys: PyArrayLike1<'py, usize> = ys.extract()?;
    let winners: PyArrayLike1<'py, u8> = winners.extract()?;
    let weights: PyArrayLike1<'py, f64> = weights.extract()?;

    match matrices(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
    ) {
        Ok((win_matrix, tie_matrix)) => {
            let matrix = win_plus_tie_matrix(
                &win_matrix.view(),
                &tie_matrix.view(),
                win_weight,
                tie_weight,
                tolerance,
            );

            match pagerank(&matrix.view(), damping, tolerance, limit) {
                Ok((scores, iterations)) => Ok((scores.into_pyarray(py).unbind(), iterations)),
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

fn alpha_distance_from_py<'py>(
    py: Python<'py>,
    distance: &Bound<'py, PyAny>,
    unique_slice: &[f64],
) -> PyResult<alpha::Distance> {
    if let Ok(matrix) = distance.extract::<PyArrayLike2<'py, f64>>() {
        let array_view = matrix.as_array();

        if array_view.nrows() != unique_slice.len() || array_view.ncols() != unique_slice.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Distance matrix must be {}x{}, got {}x{}",
                unique_slice.len(),
                unique_slice.len(),
                array_view.nrows(),
                array_view.ncols()
            )));
        }

        return Ok(alpha::Distance::CustomMatrix(array_view.to_owned()));
    }

    if distance.is_callable() {
        let py_array = PyArray1::from_slice(py, unique_slice);
        let result = distance.call1((py_array,))?;
        let matrix: PyArrayLike2<'py, f64> = result.extract()?;
        let array_view = matrix.as_array();

        if array_view.nrows() != unique_slice.len() || array_view.ncols() != unique_slice.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Distance matrix must be {}x{}, got {}x{}",
                unique_slice.len(),
                unique_slice.len(),
                array_view.nrows(),
                array_view.ncols()
            )));
        }

        return Ok(alpha::Distance::CustomMatrix(array_view.to_owned()));
    }

    let distance_str: String = distance.extract()?;
    match alpha::Distance::parse(&distance_str) {
        Ok(d) => Ok(d),
        Err(msg) => Err(UnknownDistanceError::new_err(msg)),
    }
}

#[pyfunction(name = "alpha")]
fn alpha_pyo3<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    unique_values: &Bound<'py, PyAny>,
    distance: &Bound<'py, PyAny>,
) -> PyResult<(f64, f64, f64)> {
    let codes: PyArrayLike2<'py, i64> = codes.extract()?;
    let unique_values: PyArrayLike1<'py, f64> = unique_values.extract()?;
    let codes_array = codes.as_array();
    let unique_values_array = unique_values.as_array();
    let unique_slice = unique_values_array.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("unique_values must be contiguous")
    })?;

    let distance_enum = alpha_distance_from_py(py, distance, unique_slice)?;

    match alpha::alpha_from_factorized(&codes_array, unique_slice, distance_enum) {
        Ok((alpha_value, observed, expected)) => Ok((alpha_value, observed, expected)),
        Err(msg) => {
            if msg.contains("No units have at least 2 ratings") {
                Err(InsufficientRatingsError::new_err(()))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))
            }
        }
    }
}

#[pyfunction(name = "alpha_bootstrap")]
#[pyo3(signature = (codes, unique_values, distance, n_resamples, min_resamples=1000, random_state=None))]
fn alpha_bootstrap_pyo3<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    unique_values: &Bound<'py, PyAny>,
    distance: &Bound<'py, PyAny>,
    n_resamples: i64,
    min_resamples: i64,
    random_state: Option<i64>,
) -> PyResult<(f64, f64, f64, Py<PyArray1<f64>>)> {
    let codes: PyArrayLike2<'py, i64> = codes.extract()?;
    let unique_values: PyArrayLike1<'py, f64> = unique_values.extract()?;
    let codes_array = codes.as_array();
    let unique_values_array = unique_values.as_array();
    let unique_slice = unique_values_array.as_slice().ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("unique_values must be contiguous")
    })?;

    let distance_enum = alpha_distance_from_py(py, distance, unique_slice)?;

    if n_resamples < 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "n_resamples must be a non-negative integer",
        ));
    }
    let n_resamples = usize::try_from(n_resamples).expect("non-negative i64 should fit into usize");
    if min_resamples <= 0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "min_resamples must be a positive integer",
        ));
    }
    let min_resamples = usize::try_from(min_resamples).expect("positive i64 should fit into usize");

    let random_seed = match random_state {
        Some(value) if value < 0 => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "random_state must be a non-negative integer or None",
            ));
        }
        Some(value) => Some(u64::try_from(value).expect("non-negative i64 should fit into u64")),
        None => None,
    };

    match alpha::alpha_bootstrap_from_factorized(
        &codes_array,
        unique_slice,
        distance_enum,
        n_resamples,
        min_resamples,
        random_seed,
    ) {
        Ok(result) => Ok((
            result.alpha,
            result.observed,
            result.expected,
            result.distribution.into_pyarray(py).unbind(),
        )),
        Err(msg) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)),
    }
}

#[pymodule]
#[pyo3(name = "_brzo")]
fn _brzo(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("HAS_BLAS", crate::has_blas())?;
    m.add_function(wrap_pyfunction!(matrices_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_scores_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(counting_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(average_win_rate_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(bradley_terry_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(newman_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(elo_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(eigen_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_bootstrap_pyo3, m)?)?;
    Ok(())
}
