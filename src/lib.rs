#![allow(dead_code, unused_variables, unused_imports)]
mod error;
mod fitting;
mod splines;
mod types;
mod terms;

use ndarray::Array1;
pub use error::GamError;
pub mod families;
pub use terms::Term;
pub use types::*;
pub use polars::prelude::DataFrame;

use families::Family;

#[derive(Debug)]
pub struct GeneralizedAdditiveModel {
    coefficients: Coefficients,
    covariance: CovarianceMatrix,
    terms: Vec<Term>,
}

impl GeneralizedAdditiveModel {
    pub fn fit<F: Family + 'static> (
        data: &DataFrame,
        y: &Array1<f64>,
        terms: &[Term],
        family: &F,
    ) -> Result<Self, GamError> {

        let (beta, v_beta) = fitting::fit_model(data, y, terms, family)?;
        Ok(Self {
            coefficients: beta,
            covariance: v_beta,
            terms: terms.to_vec(),
        })
    }

    pub fn posterior_samples(&self, n_samples: usize) -> Vec<Coefficients> {
        fitting::sample_posterior(&self.coefficients, &self.covariance, n_samples)
            .into_iter()
            .map(Coefficients)
            .collect()
    }
}