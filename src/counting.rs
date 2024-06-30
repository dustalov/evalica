use ndarray::{Array1, ArrayView2, Axis};

pub fn counting(m: &ArrayView2<i64>) -> Array1<i64> {
    m.sum_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::counting;

    #[test]
    fn test_counting() {
        let m = array![
            [0, 1, 2, 0, 1],
            [2, 0, 2, 1, 0],
            [1, 2, 0, 0, 1],
            [1, 2, 1, 0, 2],
            [2, 0, 1, 3, 0]
        ];

        let expected = array![4, 5, 4, 6, 6];

        let actual = counting(&m.view());

        assert_eq!(actual.len(), m.shape()[0]);

        for (a, b) in actual.iter().zip(expected.iter()) {
            assert_eq!(a, b);
        }
    }
}
