use numpy::PyArrayMethods;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike1};
use pyo3::prelude::*;
use std::convert::TryInto;

mod bradley_terry;
mod counting;
mod newman;
mod utils;

#[pyfunction]
fn py_matrices<'py>(
    py: Python<'py>,
    first: PyArrayLike1<'py, usize, AllowTypeChange>,
    second: PyArrayLike1<'py, usize, AllowTypeChange>,
    statuses: PyArrayLike1<'py, u8, AllowTypeChange>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<i64>>)> {
    let first = first.as_array().to_vec();
    let second = second.as_array().to_vec();
    let statuses = statuses
        .as_array()
        .into_iter()
        .map(|&x| x.try_into().unwrap())
        .collect();

    let (wins, ties) = utils::matrices(first, second, statuses);

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
fn py_bradley_terry(py: Python, m: &Bound<PyArray2<i64>>) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let m = unsafe { m.as_array().to_owned() };
    let (pi, iterations) = bradley_terry::bradley_terry(&m);
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

#[pymodule]
fn evalica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(py_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(py_counting, m)?)?;
    m.add_function(wrap_pyfunction!(py_bradley_terry, m)?)?;
    m.add_function(wrap_pyfunction!(py_newman, m)?)?;
    Ok(())
}
