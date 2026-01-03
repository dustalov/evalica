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
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub enum Winner {
    /// There is a tie.
    Draw = 0,
    /// The first element won.
    X = 1,
    /// The second element won.
    Y = 2,
}

impl From<u8> for Winner {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Draw,
            1 => Self::X,
            2 => Self::Y,
            _ => panic!("Invalid value: {}", value),
        }
    }
}

impl Into<u8> for Winner {
    fn into(self) -> u8 {
        match self {
            Self::Draw => 0,
            Self::X => 1,
            Self::Y => 2,
        }
    }
}
