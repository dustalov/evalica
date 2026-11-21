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

impl From<Winner> for u8 {
    fn from(val: Winner) -> Self {
        match val {
            Winner::Draw => 0,
            Winner::X => 1,
            Winner::Y => 2,
        }
    }
}
