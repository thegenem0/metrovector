use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum MvblCoercionError {
    ///
    /// Generic Coercion error for now
    ///
    CartCoercionError(String),
}

impl fmt::Display for MvblCoercionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MvblCoercionError::CartCoercionError(msg) => write!(f, "{msg}"),
        }
    }
}

impl Error for MvblCoercionError {}
