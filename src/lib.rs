use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayLike1, PyReadonlyArray2,
};
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

#[pyfunction]
fn matrices_pyo3<'py>(
    py: Python<'py>,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    ws: PyArrayLike1<'py, Winner>,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<i64>>)> {
    let (wins, ties) = utils::matrices(&xs.as_array(), &ys.as_array(), &ws.as_array(), 1, 1);

    Ok((
        wins.into_pyarray_bound(py).unbind(),
        ties.into_pyarray_bound(py).unbind(),
    ))
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
    let counts = counting::counting(
        &xs.as_array(),
        &ys.as_array(),
        &ws.as_array(),
        win_weight,
        tie_weight,
    );

    Ok(counts.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn bradley_terry_pyo3<'py>(
    py: Python,
    matrix: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let (scores, iterations) = bradley_terry::bradley_terry(&matrix.as_array(), tolerance, limit);

    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
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
    let (scores, v, iterations) = bradley_terry::newman(
        &win_matrix.as_array(),
        &tie_matrix.as_array(),
        v_init,
        tolerance,
        limit,
    );

    Ok((scores.into_pyarray_bound(py).unbind(), v, iterations))
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
    let pi = elo::elo(
        &xs.as_array(),
        &ys.as_array(),
        &ws.as_array(),
        initial,
        base,
        scale,
        k,
    );

    Ok(pi.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn eigen_pyo3<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let (scores, iterations) = linalg::eigen(
        &matrix.as_array(),
        tolerance,
        limit,
    );

    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
}

#[pyfunction]
fn pagerank_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    ws: PyArrayLike1<'py, Winner>,
    damping: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let (scores, iterations) = linalg::pagerank(
        &xs.as_array(),
        &ys.as_array(),
        &ws.as_array(),
        damping,
        win_weight,
        tie_weight,
        tolerance,
        limit,
    );

    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
}

#[pymodule]
fn evalica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
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
