use crate::bradley_terry::{bradley_terry, newman};
use crate::counting::{average_win_rate, counting};
use crate::elo::elo;
use crate::linalg::{eigen, pagerank};
use crate::utils::{matrices, win_plus_tie_matrix};
use ndarray::Array1;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = "counting")]
pub fn counting_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    win_weight: f64,
    tie_weight: f64,
) -> Result<Vec<f64>, String> {
    counting(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
        win_weight,
        tie_weight,
    )
    .map(|a| a.to_vec())
    .map_err(|e| e.to_string())
}

#[wasm_bindgen(js_name = "averageWinRate")]
pub fn average_win_rate_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    win_weight: f64,
    tie_weight: f64,
) -> Result<Vec<f64>, String> {
    average_win_rate(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
        win_weight,
        tie_weight,
    )
    .map(|a| a.to_vec())
    .map_err(|e| e.to_string())
}

#[wasm_bindgen(js_name = "bradleyTerry")]
pub fn bradley_terry_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> Result<Vec<f64>, String> {
    let (wins, ties) = matrices(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
    )
    .map_err(|e| e.to_string())?;

    let matrix = win_plus_tie_matrix(&wins.view(), &ties.view(), win_weight, tie_weight, 0.0);
    bradley_terry(&matrix.view(), tolerance, limit)
        .map(|(a, _)| a.to_vec())
        .map_err(|e| e.to_string())
}

#[wasm_bindgen(js_name = "newman")]
pub fn newman_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    v_init: f64,
    tolerance: f64,
    limit: usize,
) -> Result<Vec<f64>, String> {
    let (wins, ties) = matrices(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
    )
    .map_err(|e| e.to_string())?;

    newman(&wins.view(), &ties.view(), v_init, tolerance, limit)
        .map(|(a, _, _)| a.to_vec())
        .map_err(|e| e.to_string())
}

#[wasm_bindgen(js_name = "eigen")]
pub fn eigen_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> Result<Vec<f64>, String> {
    let (wins, ties) = matrices(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
    )
    .map_err(|e| e.to_string())?;

    let matrix = win_plus_tie_matrix(&wins.view(), &ties.view(), win_weight, tie_weight, 0.0);
    eigen(&matrix.view(), tolerance, limit)
        .map(|(a, _)| a.to_vec())
        .map_err(|e| e.to_string())
}

#[wasm_bindgen(js_name = "pagerank")]
pub fn pagerank_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    win_weight: f64,
    tie_weight: f64,
    alpha: f64,
    tolerance: f64,
    limit: usize,
) -> Result<Vec<f64>, String> {
    let (wins, ties) = matrices(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
    )
    .map_err(|e| e.to_string())?;

    let matrix = win_plus_tie_matrix(&wins.view(), &ties.view(), win_weight, tie_weight, 0.0);
    pagerank(&matrix.view(), alpha, tolerance, limit)
        .map(|(a, _)| a.to_vec())
        .map_err(|e| e.to_string())
}

#[wasm_bindgen(js_name = "elo")]
pub fn elo_wasm(
    xs: &[usize],
    ys: &[usize],
    winners: &[u8],
    weights: &[f64],
    total: usize,
    initial: f64,
    base: f64,
    scale: f64,
    k: f64,
    win_weight: f64,
    tie_weight: f64,
) -> Result<Vec<f64>, String> {
    elo(
        &Array1::from_vec(xs.to_vec()).view(),
        &Array1::from_vec(ys.to_vec()).view(),
        &Array1::from_vec(winners.to_vec()).view(),
        &Array1::from_vec(weights.to_vec()).view(),
        total,
        initial,
        base,
        scale,
        k,
        win_weight,
        tie_weight,
    )
    .map(|a| a.to_vec())
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;


    #[wasm_bindgen_test]
    fn test_counting() {
        let xs = vec![0, 1, 2];
        let ys = vec![1, 2, 0];
        let winners = vec![1, 1, 1]; // X wins
        let weights = vec![1.0, 1.0, 1.0];
        let result = counting_wasm(&xs, &ys, &winners, &weights, 3, 1.0, 0.5).unwrap();
        assert_eq!(result.len(), 3);
        for &score in &result {
            assert!(score > 0.0);
        }
    }

    #[wasm_bindgen_test]
    fn test_bradley_terry() {
        let xs = vec![0, 1, 2];
        let ys = vec![1, 2, 0];
        let winners = vec![1, 1, 1];
        let weights = vec![1.0, 1.0, 1.0];
        let result =
            bradley_terry_wasm(&xs, &ys, &winners, &weights, 3, 1.0, 0.5, 1e-6, 100).unwrap();
        assert_eq!(result.len(), 3);
    }

    #[wasm_bindgen_test]
    fn test_elo() {
        let xs = vec![0, 1, 2];
        let ys = vec![1, 2, 0];
        let winners = vec![1, 1, 1];
        let weights = vec![1.0, 1.0, 1.0];
        let result = elo_wasm(
            &xs, &ys, &winners, &weights, 3, 1500.0, 10.0, 400.0, 32.0, 1.0, 0.5,
        )
        .unwrap();
        assert_eq!(result.len(), 3);
    }
}
