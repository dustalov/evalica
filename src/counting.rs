use std::ops::AddAssign;

use ndarray::{Array1, ArrayView1, ErrorKind, ShapeError};
use num_traits::Num;

use crate::{match_lengths, Winner};

pub fn counting<A: Num + Copy + AddAssign>(
    xs: &ArrayView1<usize>,
    ys: &ArrayView1<usize>,
    ws: &ArrayView1<Winner>,
    win_weight: A,
    tie_weight: A,
) -> Result<Array1<A>, ShapeError> {
    match_lengths!(xs.len(), ys.len(), ws.len());

    if xs.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let n = 1 + std::cmp::max(*xs.iter().max().unwrap(), *ys.iter().max().unwrap());

    let mut scores = Array1::zeros(n);

    for ((x, y), &ref w) in xs.iter().zip(ys.iter()).zip(ws.iter()) {
        match w {
            Winner::X => scores[*x] += win_weight,
            Winner::Y => scores[*y] += win_weight,
            Winner::Draw => {
                scores[*x] += tie_weight;
                scores[*y] += tie_weight;
            }
            _ => {}
        }
    }

    Ok(scores)
}

#[cfg(test)]
mod tests {
    use ndarray::{array, ArrayView1};

    use crate::utils;

    use super::counting;

    #[test]
    fn test_counting() {
        let xs = ArrayView1::from(&utils::fixtures::XS);
        let ys = ArrayView1::from(&utils::fixtures::YS);
        let ws = ArrayView1::from(&utils::fixtures::WS);

        let expected = array![1.5, 3.0, 3.0, 4.5, 4.0];

        let actual = counting(&xs.view(), &ys.view(), &ws.view(), 1.0, 0.5).unwrap();

        assert_eq!(actual, expected);
    }
}
