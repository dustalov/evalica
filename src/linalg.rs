use std::collections::HashMap;

use ndarray::{Array1, ArrayView1};

use crate::Winner;

#[derive(Clone, Debug)]
struct Edge {
    x: usize,
    y: usize,
    weight: f64,
}

fn compute_weights(
    nodes: &[usize],
    out_edges_map: &HashMap<usize, Vec<Edge>>,
) -> HashMap<usize, f64> {
    nodes
        .iter()
        .map(|&n| {
            let weight_sum = out_edges_map
                .get(&n)
                .map_or(0.0, |edges| edges.iter().map(|e| e.weight).sum());
            (n, weight_sum)
        })
        .collect()
}

fn compute_sinks(nodes: &[usize], out_edges_map: &HashMap<usize, Vec<Edge>>) -> Vec<usize> {
    nodes
        .iter()
        .filter(|&&n| out_edges_map.get(&n).map_or(true, |edges| edges.is_empty()))
        .cloned()
        .collect()
}

fn compute_edges(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    win_weight: f64,
    tie_weight: f64,
) -> (HashMap<usize, Vec<Edge>>, HashMap<usize, Vec<Edge>>) {
    let mut out_edges_map: HashMap<usize, Vec<Edge>> = HashMap::new();
    let mut in_edges_map: HashMap<usize, Vec<Edge>> = HashMap::new();
    let mut edges = Vec::new();

    for (&x, (&y, &ref w)) in xs.iter().zip(ys.iter().zip(ws.iter())) {
        match w {
            Winner::X => edges.push(Edge {
                x: x,
                y: y,
                weight: win_weight,
            }),
            Winner::Y => edges.push(Edge {
                x: y,
                y: x,
                weight: win_weight,
            }),
            Winner::Draw => {
                edges.push(Edge {
                    x: x,
                    y: y,
                    weight: tie_weight,
                });
                edges.push(Edge {
                    x: y,
                    y: x,
                    weight: tie_weight,
                });
            }
            _ => {}
        }
    }

    for &node in xs.iter().chain(ys.iter()) {
        in_edges_map.insert(node, Vec::new());
        out_edges_map.insert(node, Vec::new());
    }

    for edge in edges.iter() {
        out_edges_map.get_mut(&edge.x).unwrap().push(edge.clone());
        in_edges_map.get_mut(&edge.y).unwrap().push(edge.clone());
    }

    (out_edges_map, in_edges_map)
}

pub fn pagerank(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    damping: f64,
    win_weight: f64,
    tie_weight: f64,
    tolerance: f64,
    limit: usize,
) -> (Array1<f64>, usize) {
    assert_eq!(
        xs.len(),
        ys.len(),
        "first and second length mismatch: {} vs. {}",
        xs.len(),
        ys.len()
    );

    assert_eq!(
        xs.len(),
        ws.len(),
        "first and status length mismatch: {} vs. {}",
        xs.len(),
        ws.len()
    );

    if xs.is_empty() {
        return (Array1::zeros(0), 0);
    }

    let (out_edges_map, in_edges_map) = compute_edges(xs, ys, ws, win_weight, tie_weight);
    let nodes: Vec<usize> = out_edges_map.keys().cloned().collect();
    let n = nodes.len();

    let mut scores = vec![1.0 / n as f64; n];
    let edge_weights = compute_weights(&nodes, &out_edges_map);
    let sinks = compute_sinks(&nodes, &out_edges_map);

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < limit {
        converged = true;
        iterations += 1;

        let mut scores_new = scores.clone();
        let sinkrank: f64 = sinks.iter().map(|&n| scores[n]).sum();

        for &node in &nodes {
            let teleportation = (1.0 - damping) / n as f64;
            let spreading = damping * sinkrank / n as f64;

            scores_new[node] =
                in_edges_map[&node]
                    .iter()
                    .fold(teleportation + spreading, |r, e| {
                        let q = if e.x == node { e.y } else { e.x };
                        let weight = e.weight / edge_weights[&q];
                        r + damping * scores[q] * weight
                    });

            if ((scores_new[node] - scores[node]) / scores[node]).abs() >= tolerance {
                converged = false;
            }
        }

        scores = scores_new;
    }

    (Array1::from_vec(scores), iterations)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn test_pagerank() {
        let xs = array![0, 1, 2, 0];
        let ys = array![1, 2, 3, 3];
        let ws = array![Winner::X, Winner::X, Winner::X, Winner::Y];
        let tolerance = 1e-4;

        let expected = array![0.25, 0.25, 0.25, 0.25];

        let (actual, iterations) = pagerank(
            &xs.view(),
            &ys.view(),
            &ws.view(),
            0.85,
            1.0,
            0.5,
            tolerance,
            100,
        );

        assert!(iterations > 0);

        for (left, right) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(left, right, epsilon = tolerance);
        }
    }
}
