use thiserror::Error;

use crate::mvf_fbs::VectorType;

pub type Result<T> = std::result::Result<T, MvfError>;

#[derive(Error, Debug)]
pub enum MvfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported version: got {got}, expected {expected}")]
    UnsupportedVersion { got: u16, expected: u16 },

    #[error("Vector space '{0}' not found")]
    VectorSpaceNotFound(String),

    #[error("Index out of bounds: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid vector type: expected {expected:?}, got {actual:?}")]
    InvalidVectorType {
        expected: VectorType,
        actual: VectorType,
    },
    #[error("Corrupted data: {0}")]
    CorruptedData(String),

    #[error("Extension error: {0}")]
    Extension(String),

    #[error("Build error: {0}")]
    Build(String),
}

impl MvfError {
    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::InvalidFormat(msg.into())
    }

    pub fn corrupted_data(msg: impl Into<String>) -> Self {
        Self::CorruptedData(msg.into())
    }

    pub fn build_error(msg: impl Into<String>) -> Self {
        Self::Build(msg.into())
    }
}
