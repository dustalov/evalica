use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayLike1, PyReadonlyArray2,
};
use pyo3::prelude::*;

mod bradley_terry;
mod counting;
mod elo;
mod linalg;
mod newman;
mod utils;

#[pyclass]
#[repr(u8)]
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Won,
    Lost,
    Tied,
    Skipped,
}

unsafe impl Element for Status {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        numpy::dtype_bound::<u8>(py)
    }
}

#[pyfunction]
fn py_matrices<'py>(
    py: Python<'py>,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    rs: PyArrayLike1<'py, Status>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<i64>>)> {
    let (wins, ties) = utils::matrices(&xs.as_array(), &ys.as_array(), &rs.as_array());

    Ok((
        wins.into_pyarray_bound(py).unbind(),
        ties.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
fn py_counting<'py>(py: Python, m: PyReadonlyArray2<'py, i64>) -> PyResult<Py<PyArray1<i64>>> {
    let counts = counting::counting(&m.as_array());

    Ok(counts.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn py_bradley_terry<'py>(
    py: Python,
    m: PyReadonlyArray2<'py, i64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let (scores, iterations) = bradley_terry::bradley_terry(&m.as_array(), tolerance, limit);

    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
}

#[pyfunction]
fn py_newman<'py>(
    py: Python,
    m: PyReadonlyArray2<'py, i64>,
    seed: u64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let (pi, iterations) = newman::newman(&m.as_array(), seed, tolerance, limit);

    Ok((pi.into_pyarray_bound(py).unbind(), iterations))
}

#[pyfunction]
fn py_elo<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    rs: PyArrayLike1<'py, Status>,
    r: f64,
    k: u64,
    s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let pi = elo::elo(&xs.as_array(), &ys.as_array(), &rs.as_array(), r, k, s);

    Ok(pi.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn py_eigen<'py>(py: Python, m: PyReadonlyArray2<'py, f64>) -> PyResult<Py<PyArray1<f64>>> {
    let eigen = linalg::eigen(&m.as_array());

    Ok(eigen.into_pyarray_bound(py).unbind())
}

#[pymodule]
fn evalica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(py_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(py_counting, m)?)?;
    m.add_function(wrap_pyfunction!(py_bradley_terry, m)?)?;
    m.add_function(wrap_pyfunction!(py_newman, m)?)?;
    m.add_function(wrap_pyfunction!(py_elo, m)?)?;
    m.add_function(wrap_pyfunction!(py_eigen, m)?)?;
    m.add_class::<Status>()?;
    Ok(())
}
