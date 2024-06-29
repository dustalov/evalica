use std::convert::TryInto;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayLike1};
use numpy::PyArrayMethods;
use pyo3::prelude::*;

use crate::utils::Status;

mod bradley_terry;
mod counting;
mod elo;
mod newman;
mod utils;

fn transform_statuses(statuses: &PyArrayLike1<u8>) -> Vec<Status> {
    let statuses = statuses
        .as_array()
        .into_iter()
        .map(|&x| x.try_into().unwrap())
        .collect();
    statuses
}

#[pyfunction]
fn py_matrices<'py>(
    py: Python<'py>,
    first: PyArrayLike1<'py, usize>,
    second: PyArrayLike1<'py, usize>,
    statuses: PyArrayLike1<'py, u8>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<i64>>)> {
    let first = first.as_array().to_vec();
    let second = second.as_array().to_vec();
    let statuses = transform_statuses(&statuses);

    let (wins, ties) = utils::matrices(&first, &second, &statuses);

    Ok((
        wins.into_pyarray_bound(py).unbind(),
        ties.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
fn py_counting(py: Python, m: &Bound<PyArray2<i64>>) -> PyResult<Py<PyArray1<i64>>> {
    let m = unsafe { m.as_array().to_owned() };
    let counts = counting::counting(&m);
    Ok(counts.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn py_bradley_terry(
    py: Python,
    m: &Bound<PyArray2<i64>>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let m = unsafe { m.as_array().to_owned() };
    let (pi, iterations) = bradley_terry::bradley_terry(&m, tolerance, limit);
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
    let (pi, iterations) = newman::newman(&m, seed, tolerance, limit);
    Ok((pi.into_pyarray_bound(py).unbind(), iterations))
}

#[pyfunction]
fn py_elo<'py>(
    py: Python,
    first: PyArrayLike1<'py, usize>,
    second: PyArrayLike1<'py, usize>,
    statuses: PyArrayLike1<'py, u8>,
    r: f64,
    k: u64,
    s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let first = first.as_array().to_vec();
    let second = second.as_array().to_vec();
    let statuses = transform_statuses(&statuses);

    let pi = elo::elo(&first, &second, &statuses, r, k, s);
    Ok(pi.into_pyarray_bound(py).unbind())
}

#[pymodule]
fn evalica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(py_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(py_counting, m)?)?;
    m.add_function(wrap_pyfunction!(py_bradley_terry, m)?)?;
    m.add_function(wrap_pyfunction!(py_newman, m)?)?;
    m.add_function(wrap_pyfunction!(py_elo, m)?)?;
    Ok(())
}
