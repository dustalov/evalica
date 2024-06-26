use ndarray::{Array1, Array2, Axis};

pub fn counting(m: &Array2<i64>) -> Array1<i64> {
    m.sum_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};

    use super::counting;

    #[test]
    fn test_counting() {
        let m: Array2<i64> = array![
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0]
        ];

        let p = counting(&m);

        assert_eq!(p.len(), m.shape()[0]);

        let expected_p = array![4, 5, 4, 6, 6];

        for (a, b) in p.iter().zip(expected_p.iter()) {
            assert!(a == b);
        }
    }
}
