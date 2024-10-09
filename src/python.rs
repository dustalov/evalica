use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayLike1};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::bradley_terry::{bradley_terry, newman};
use crate::counting::{average_win_rate, counting};
use crate::elo::elo;
use crate::linalg::{eigen, pagerank};
use crate::utils::{matrices, nan_to_num, pairwise_scores, win_plus_tie_matrix};
use crate::Winner;

#[pymethods]
impl Winner {
    #[new]
    fn new() -> Self {
        Winner::Draw
    }

    fn __hash__(&self) -> u64 {
        let value: u8 = self.clone().into();
        value.into()
    }

    fn __getstate__(&self) -> PyResult<u8> {
        Ok(self.clone().into())
    }

    fn __setstate__(&mut self, state: u8) -> PyResult<()> {
        *self = Winner::from(state);
        Ok(())
    }
}

unsafe impl Element for Winner {
    const IS_COPY: bool = true;

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        Clone::clone(self)
    }

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        numpy::dtype_bound::<u8>(py)
    }
}

create_exception!(evalica, LengthMismatchError, PyValueError);

#[pyfunction]
fn matrices_pyo3<'py>(
    py: Python<'py>,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    match matrices(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
    ) {
        Ok((wins, ties)) => Ok((
            wins.into_pyarray_bound(py).unbind(),
            ties.into_pyarray_bound(py).unbind(),
        )),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn pairwise_scores_pyo3<'py>(
    py: Python,
    scores: PyArrayLike1<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let pairwise = pairwise_scores(&scores.as_array());

    Ok(pairwise.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn counting_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    match counting(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
        win_weight,
        tie_weight,
    ) {
        Ok(scores) => Ok(scores.into_pyarray_bound(py).unbind()),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn average_win_rate_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    match average_win_rate(
        &xs.as_array(),
        &ys.as_array(),
        &winners.as_array(),
        &weights.as_array(),
        total,
        win_weight,
        tie_weight,
    ) {
        Ok(scores) => Ok(scores.into_pyarray_bound(py).unbind()),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn bradley_terry_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
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
                Ok((scores, iterations)) => {
                    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
                }
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn newman_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    v_init: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, f64, usize)> {
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
                    Ok((scores.into_pyarray_bound(py).unbind(), v, iterations))
                }
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn elo_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    initial: f64,
    base: f64,
    scale: f64,
    k: f64,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    match elo(
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
    ) {
        Ok(scores) => Ok(scores.into_pyarray_bound(py).unbind()),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn eigen_pyo3<'py>(
    py: Python<'py>,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
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
                Ok((scores, iterations)) => {
                    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
                }
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn pagerank_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    winners: PyArrayLike1<'py, Winner>,
    weights: PyArrayLike1<'py, f64>,
    total: usize,
    damping: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
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
                Ok((scores, iterations)) => {
                    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
                }
                Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
            }
        }
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pymodule]
fn evalica(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "LengthMismatchError",
        py.get_type_bound::<LengthMismatchError>(),
    )?;
    m.add_function(wrap_pyfunction!(matrices_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_scores_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(counting_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(average_win_rate_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(bradley_terry_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(newman_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(elo_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(eigen_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank_pyo3, m)?)?;
    m.add_class::<Winner>()?;
    Ok(())
}
