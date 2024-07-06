use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescr, PyArrayLike1, PyReadonlyArray2,
};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;

mod bradley_terry;
mod counting;
mod elo;
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
    let (wins, ties) = utils::matrices(&xs.as_array(), &ys.as_array(), &ws.as_array());

    Ok((
        wins.into_pyarray_bound(py).unbind(),
        ties.into_pyarray_bound(py).unbind(),
    ))
}

#[pyfunction]
fn counting_pyo3<'py>(py: Python, m: PyReadonlyArray2<'py, i64>) -> PyResult<Py<PyArray1<i64>>> {
    let counts = counting::counting(&m.as_array());

    Ok(counts.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn bradley_terry_pyo3<'py>(
    py: Python,
    m: PyReadonlyArray2<'py, f64>,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, usize)> {
    let (scores, iterations) = bradley_terry::bradley_terry(&m.as_array(), tolerance, limit);

    Ok((scores.into_pyarray_bound(py).unbind(), iterations))
}

#[pyfunction]
fn newman_pyo3<'py>(
    py: Python,
    w: PyReadonlyArray2<'py, f64>,
    t: PyReadonlyArray2<'py, f64>,
    v_init: f64,
    tolerance: f64,
    limit: usize,
) -> PyResult<(Py<PyArray1<f64>>, f64, usize)> {
    let (scores, v, iterations) =
        bradley_terry::newman(&w.as_array(), &t.as_array(), v_init, tolerance, limit);

    Ok((scores.into_pyarray_bound(py).unbind(), v, iterations))
}

#[pyfunction]
fn elo_pyo3<'py>(
    py: Python,
    xs: PyArrayLike1<'py, usize>,
    ys: PyArrayLike1<'py, usize>,
    ws: PyArrayLike1<'py, Winner>,
    r: f64,
    k: u64,
    s: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let pi = elo::elo(&xs.as_array(), &ys.as_array(), &ws.as_array(), r, k, s);

    Ok(pi.into_pyarray_bound(py).unbind())
}

#[pyfunction]
fn eigen_pyo3<'py>(py: Python<'py>, m: PyReadonlyArray2<'py, f64>) -> PyResult<Py<PyArray1<f64>>> {
    /*
    I found this approach simpler than setting up BLAS
    for multiple platforms which is required by ndarray-linalg.
    We need to re-implement eigenvector centrality
    on our own instead of making a round-trip to Python.
    */

    let np = py.import_bound("numpy")?;
    let globals = [("np", np)].into_py_dict_bound(py);
    let locals = [("m", m.as_unbound())].into_py_dict_bound(py);

    let eigen = py
        .eval_bound(
            "np.linalg.eigh(m).eigenvalues",
            Some(&globals),
            Some(&locals),
        )?
        .downcast_into::<PyArray1<f64>>()?;

    Ok(eigen.unbind())
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
    m.add_class::<Winner>()?;
    Ok(())
}
