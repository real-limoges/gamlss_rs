use bspline;
use ndarray::{Array1, Array2};
use super::types::{Matrix, Vector};


pub fn create_basis_matrix(x: &Vector, n_splines: usize, degree: usize) -> Matrix {
    // creates a basis matrix of bsplines as the foundation for our models
    let n_obs = x.len();
    let knots = select_knots(x, n_splines - degree + 1, degree);

    let spline = bspline::BSpline::new(degree, knots);

    let mut basis_matrix = Matrix::zeros((n_obs, n_splines));
    for (i, &x_i) in x.iter().enumerate() {
        let basis_evals = spline.eval_all(x_i);
        for (j, &val) in basis_evals.iter().enumerate() {
            basis_matrix[(i, j)] = val;
        }
    }
    basis_matrix
}

fn select_knots(x: &Vector, n_knots: usize, degree: usize) -> Vec<f64> {
    //  selects the correct number of knots for smooth terms

    let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let mut interior_knots = Vec::with_capacity(n_knots.saturating_sub(2));
    let mut sorted_x = x.to_vec();
    sorted_x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    for i in 1..=n_knots.saturating_sub(2) {
        let quantile = (i as f64) / ((n_knots - 1) as f64);
        let index = (quantile * (sorted_x.len() - 1) as f64) as usize;
        interior_knots.push(sorted_x[index]);
    }

    // I'm sure there's a better way to push these in, but the there's only, maybe 40?, knots
    let mut knots = Vec::new();
    for _ in 0..=degree {
        knots.push(min_val);
    }
    knots.append(&mut interior_knots);
    for _ in 0..=degree {
        knots.push(max_val);
    }
    knots
}

// P----- Add Penalty Matrices

pub fn create_penalty_matrix(n_splines: usize, order: usize) -> Matrix {
    // I only implemented two types of penalties. First and Second Order.

    let n_rows_d = n_splines.saturating_sub(order);
    if n_rows_d == 0 {
        return Matrix::zeros((n_splines, n_splines));
    }

    let mut d_matrix = Matrix::zeros((n_rows_d, n_splines));
    match order {
        1 =>
            for i in 0..n_rows_d {
                d_matrix[[i,i] = 1.0];
                d_matrix[[i, i+1]] = -1.0;
            }
        _ =>
        // this is the second order penalty. I gave it to everything that isn't 1
        for i in 0..n_rows_d {
            d_matrix[[i,i]] = 1.0;
            d_matrix[[i,i+1]] = -2.0;
            d_matrix[[i,i+2]] = 1.0;
        }
    }
    d_matrix.t().dot(&d_matrix)
}

pub fn kronecker_product(a: &Matrix, b: &Matrix) -> Matrix {
    // A little bit of linear algebra magic
    let (m,n) = a.dims();
    let (p,q) = b.dims();
    let mut c = Matrix::zeros((m*p, p*q));

    for i in 0..m {
        for j in 0..n {
            let a_scalar = a[[i,j]];
            let mut block = c.slice_mut(s![i*p..(i+1)*p, j*q..(j+1)*q]);
            block.assign(&(b*a_scalar));
        }
    }
    c
}