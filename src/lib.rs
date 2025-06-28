#[cfg(feature = "python")]
use pyo3::prelude::pyclass;

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

/// The outcome of the pairwise comparison.
#[cfg_attr(feature = "python", pyclass(module = "evalica", eq, eq_int))]
#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Winner {
    /// The first element won.
    X,
    /// The second element won.
    Y,
    /// There is a tie.
    Draw,
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
