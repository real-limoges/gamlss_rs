use thiserror::Error;
use polars::prelude::PolarsError;

#[derive(Debug, Error)]
pub enum GamError {
    #[error(Optimization failed: {0})]
    Optimization(String),

    #[error("Linear algebra error: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    #[error("Array shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),

    #[error("PIRLS algorithm failed to converge after {0} iterations")]
    Convergence(usize),

    #[error("Invalid input: {0}")]
    Input(String),

    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
}

impl From<argmin::core::Error> for GamError {
    fn from(e: argmin::core::Error) -> Self {
        GamError::Optimization(e.to_string())
    }
}