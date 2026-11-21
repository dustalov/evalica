use ndarray::ArrayView1;

/// Implements Krippendorff's alpha for inter-rater reliability.
pub mod alpha;
/// Implements the Bradley–Terry model and related methods.
pub mod bradley_terry;
/// Implements counting-based rating methods.
pub mod counting;
/// Implements the Elo rating system.
pub mod elo;
/// Implements linear algebra-based rating methods.
pub mod linalg;
#[cfg(feature = "python")]
mod python;
/// Contains utility functions and macros.
pub mod utils;
#[cfg(feature = "wasm")]
pub mod wasm;

/// The outcome of the pairwise comparison.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Winner {
    /// There is a tie.
    Draw = 0,
    /// The first element won.
    X = 1,
    /// The second element won.
    Y = 2,
}

impl TryFrom<u8> for Winner {
    type Error = &'static str;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Draw),
            1 => Ok(Self::X),
            2 => Ok(Self::Y),
            _ => Err("invalid winner value"),
        }
    }
}

impl Winner {
    /// Reinterprets the winner codes as winners without copying.
    ///
    /// # Errors
    ///
    /// Returns an error if any code is not a valid winner.
    pub fn view_from<'a>(codes: &ArrayView1<'a, u8>) -> Result<ArrayView1<'a, Self>, &'static str> {
        if codes.iter().any(|&code| Self::try_from(code).is_err()) {
            return Err("invalid winner value");
        }

        // SAFETY: Winner is repr(u8) and every code is a valid discriminant.
        Ok(unsafe { codes.raw_view().cast::<Self>().deref_into_view() })
    }
}

impl From<Winner> for u8 {
    fn from(val: Winner) -> Self {
        match val {
            Winner::Draw => 0,
            Winner::X => 1,
            Winner::Y => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Winner;
    use ndarray::{array, s};

    #[test]
    fn test_view_from() {
        let codes = array![1u8, 0, 2, 7, 2];

        let winners = Winner::view_from(&codes.slice(s![..3])).unwrap();
        assert_eq!(winners, array![Winner::X, Winner::Draw, Winner::Y]);

        let strided = Winner::view_from(&codes.slice(s![..;2])).unwrap();
        assert_eq!(strided, array![Winner::X, Winner::Y, Winner::Y]);

        let invalid = Winner::view_from(&codes.slice(s![1..;2])).unwrap_err();
        assert_eq!(invalid, "invalid winner value");
    }
}
