#[cfg(feature = "python")]
use pyo3::prelude::pyclass;

mod bradley_terry;
mod counting;
mod elo;
mod linalg;
#[cfg(feature = "python")]
mod python;
mod utils;

#[cfg_attr(feature = "python", pyclass(module = "evalica", eq, eq_int))]
#[repr(u8)]
#[derive(Clone, Debug, PartialEq, Hash)]
pub enum Winner {
    X,
    Y,
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
