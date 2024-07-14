use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayLike1, PyReadonlyArray2,
};
use pyo3::create_exception;
use pyo3::prelude::*;

mod bradley_terry;
mod counting;
mod elo;
mod linalg;
mod utils;

#[pyclass]
#[repr(u8)]
#[derive(Clone, Debug, PartialEq)]
pub enum Winner {
    X,
    Y,
    Draw,
    Ignore,
}

unsafe impl Element for Winner {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        numpy::dtype_bound::<u8>(py)
    }
}

create_exception!(evalica, LengthMismatchError, pyo3::exceptions::PyException);

#[pyfunction]
fn matrices_pyo3<'py>(
    py: Python<'py>,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    ws: PyArrayLike1<'py, Winner>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<i64>>)> {
    let result = utils::matrices(&xs.as_array(), &ys.as_array(), &ws.as_array(), 1, 1);

    match result {
        Ok((wins, ties)) => Ok((
            wins.into_pyarray_bound(py).unbind(),
            ties.into_pyarray_bound(py).unbind(),
        )),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn counting_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    ws: PyArrayLike1<'py, Winner>,
    win_weight: f64,
    tie_weight: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = counting::counting(
        &xs.as_array(),
        &ys.as_array(),
        &ws.as_array(),
        win_weight,
        tie_weight,
    );

    match result {
        Ok(scores) => Ok(scores.into_pyarray_bound(py).unbind()),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn bradley_terry_pyo3<'py>(
    py: Python,
    matrix: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let result = bradley_terry::bradley_terry(&matrix.as_array(), tolerance, limit);

    match result {
        Ok((scores, iterations)) => Ok((scores.into_pyarray_bound(py).unbind(), iterations)),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn newman_pyo3<'py>(
    py: Python,
    win_matrix: PyReadonlyArray2<'py, f64>,
    tie_matrix: PyReadonlyArray2<'py, f64>,
    v_init: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, f64, usize)> {
    let result = bradley_terry::newman(
        &win_matrix.as_array(),
        &tie_matrix.as_array(),
        v_init,
        tolerance,
        limit,
    );

    match result {
        Ok((scores, v, iterations)) => Ok((scores.into_pyarray_bound(py).unbind(), v, iterations)),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn elo_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    ws: PyArrayLike1<'py, Winner>,
    initial: f64,
    base: f64,
    scale: f64,
    k: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = elo::elo(
        &xs.as_array(),
        &ys.as_array(),
        &ws.as_array(),
        initial,
        base,
        scale,
        k,
    );

    match result {
        Ok(scores) => Ok(scores.into_pyarray_bound(py).unbind()),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn eigen_pyo3<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let result = linalg::eigen(&matrix.as_array(), tolerance, limit);

    match result {
        Ok((scores, iterations)) => Ok((scores.into_pyarray_bound(py).unbind(), iterations)),
        Err(_) => Err(LengthMismatchError::new_err("mismatching input shapes")),
    }
}

#[pyfunction]
fn pagerank_pyo3<'py>(
    py: Python,
    win_matrix: PyReadonlyArray2<'py, f64>,
    tie_matrix: PyReadonlyArray2<'py, f64>,
    damping: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let result = linalg::pagerank(
        &win_matrix.as_array(),
        &tie_matrix.as_array(),
        damping,
        win_weight,
        tie_weight,
        tolerance,
        limit,
    );

    match result {
        Ok((scores, iterations)) => Ok((scores.into_pyarray_bound(py).unbind(), iterations)),
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
    m.add_function(wrap_pyfunction!(counting_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(bradley_terry_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(newman_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(elo_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(eigen_pyo3, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank_pyo3, m)?)?;
    m.add_class::<Winner>()?;
    Ok(())
}
