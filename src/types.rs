use ndarray::{Array1, Array2};
use std::ops::{Add, Deref, DerefMut, Sub};
use argmin::core::ArgminError;
use argmin_math::{ArgminAdd, ArgminSub, ArgminDot, ArgminScaledAdd, ArgminScaledSub, ArgminL1Norm};

// ----- Mnemonics because the ndarray names stink
pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

// ----- Newtypes for Safety (Vectors)
#[derive(Debug, Clone)]
pub struct Coefficients(pub Vector);

#[derive(Debug, Clone)]
pub struct LogLambdas(pub Vector);

// ----- Newtypes for Safety (Matrices)
#[derive(Debug, Clone)]
pub struct ModelMatrix(pub Matrix);

#[derive(Debug, Clone)]
pub struct PenaltyMatrix(pub Matrix);

#[derive(Debug, Clone)]
pub struct CovarianceMatrix(pub Matrix);


// ----- Impls for LogLambdas (I need to do a bunch of them for argmin_math)
impl Deref for LogLambdas {
    type Target = Vector;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for LogLambdas {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ArgminSub<LogLambdas, LogLambdas> for LogLambdas {
    fn sub(&self, other: &LogLambdas) -> LogLambdas { LogLambdas((&self.0).sub(&other.0)) }
}
impl ArgminAdd<LogLambdas, LogLambdas> for LogLambdas {
    fn add(&self, other: &LogLambdas) -> LogLambdas { LogLambdas((&self.0).add(&other.0)) }
}
impl ArgminDot<LogLambdas, f64> for LogLambdas {
    fn dot(&self, other: &LogLambdas) -> f64 { (&self.0).dot(&other.0) }
}
impl ArgminScaledAdd<LogLambdas, f64, LogLambdas> for LogLambdas {
    fn scaled_add(&self, scalar: &f64, other: &LogLambdas) -> LogLambdas {
        LogLambdas(&self.0 + (*scalar * &other.0))
    }
}
impl ArgminScaledSub<LogLambdas, f64, LogLambdas> for LogLambdas {
    fn scaled_sub(&self, scalar: &f64, other: &LogLambdas) -> LogLambdas {
        LogLambdas(&self.0 - (*scalar * &other.0))
    }
}
// ----- Impls for everything else
impl Deref for Coefficients {
    type Target = Vector;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Coefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}



impl Deref for ModelMatrix {
    type Target = Matrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ModelMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for PenaltyMatrix {
    type Target = Matrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PenaltyMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for CovarianceMatrix {
    type Target = Matrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for CovarianceMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}