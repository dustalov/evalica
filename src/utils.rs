use ndarray::Array2;
use std::convert::TryFrom;

#[repr(u8)]
pub enum Status {
    Won,
    Lost,
    Tied,
    Skipped,
}

impl TryFrom<u8> for Status {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Status::Won),
            1 => Ok(Status::Lost),
            2 => Ok(Status::Tied),
            3 => Ok(Status::Skipped),
            _ => Err(()),
        }
    }
}

pub fn matrices(
    first: Vec<usize>,
    second: Vec<usize>,
    status: Vec<Status>,
) -> (Array2<i64>, Array2<i64>) {
    let n = first.len();

    let mut wins = Array2::zeros((n, n));
    let mut ties = Array2::zeros((n, n));

    for i in 0..n {
        match status[i] {
            Status::Won => {
                wins[[first[i], second[i]]] += 1;
            }
            Status::Lost => {
                wins[[second[i], first[i]]] += 1;
            }
            Status::Tied => {
                ties[[first[i], second[i]]] += 1;
                ties[[second[i], first[i]]] += 1;
            }
            Status::Skipped => {}
        }
    }

    (wins, ties)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::{matrices, Status};

    #[test]
    fn test_matrices() {
        let first = vec![0, 1, 2, 3];
        let second = vec![1, 2, 3, 4];
        let status = vec![Status::Won, Status::Lost, Status::Tied, Status::Skipped];

        let (wins, ties) = matrices(first, second, status);

        assert_eq!(
            wins,
            array![[0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0],]
        );

        assert_eq!(
            ties,
            array![[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],]
        );
    }
}
