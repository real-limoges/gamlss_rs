#[derive(Debug, Clone)]
pub enum Term {
    Intercept,
    // non-penalized linear term
    Linear {
        col_name: String,
    },
    // smooth portion for one term
    Smooth(Smooth),
}

#[derive(Debug, Clone)]
pub enum Smooth {
    PSpline1D {
        col_name: String,
        n_splines: usize,
        degree: usize,
        penalty_order: usize,
    },
    TensorProduct {
        col_name_1: String,
        n_splines_1: usize,
        penalty_order_1: usize,

        col_name_2: String,
        n_splines_2: usize,
        penalty_order_2: usize,

        // I'm just forcing them to have the same degree
        degree: usize,
    },
    RandomEffect {
        col_name: String,
    },
}