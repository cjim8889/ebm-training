// Stan model for Lennard-Jones 13 particles with harmonic potential

functions {
  // Lennard-Jones potential for a single pair
  real lj_potential(real r_sq, real sigma_sq, real epsilon) {
    real r6_inv = (sigma_sq / r_sq)^3;
    real r12_inv = r6_inv^2;
    return epsilon * (r12_inv - 2.0 * r6_inv);
  }
}

data {
  int<lower=1> N;              // Number of particles
  int<lower=1> n_spatial_dim;  // Number of spatial dimensions
  int<lower=1> D;              // Total dimension (N * n_spatial_dim)
}

transformed data {
  real sigma = 1.0;
  real epsilon = 1.0;
  real c_harmonic = 0.5;
  real sigma_sq = square(sigma);
}

parameters {
  vector[D] x; // Flat vector of particle positions (x1_1, x1_2, ..., xN_spatial_dim)
}

model {
  real total_lj_energy = 0.0;
  real harmonic_energy = 0.0;
  vector[n_spatial_dim] x_com = rep_vector(0.0, n_spatial_dim); // Center of mass

  // Reshape x into a matrix or array for easier indexing
  // Stan doesn't have direct matrix reshaping in model block, use indexing
  // Calculate COM first
  for (p in 1:N) {
    int start_idx = (p - 1) * n_spatial_dim + 1;
    int end_idx = p * n_spatial_dim;
    vector[n_spatial_dim] pos_p = x[start_idx:end_idx];
    x_com += pos_p;
  }
  x_com /= N;

  // Calculate LJ and Harmonic potential
  for (i in 1:(N - 1)) {
    int start_idx_i = (i - 1) * n_spatial_dim + 1;
    int end_idx_i = i * n_spatial_dim;
    vector[n_spatial_dim] pos_i = x[start_idx_i:end_idx_i];

    // Harmonic contribution for particle i
    harmonic_energy += dot_self(pos_i - x_com);

    for (j in (i + 1):N) {
      int start_idx_j = (j - 1) * n_spatial_dim + 1;
      int end_idx_j = j * n_spatial_dim;
      vector[n_spatial_dim] pos_j = x[start_idx_j:end_idx_j];

      // Calculate squared distance to avoid sqrt
      real r_sq = squared_distance(pos_i, pos_j);

      // Add LJ potential contribution
      // Add a small epsilon to avoid division by zero if particles overlap exactly
      total_lj_energy += lj_potential(r_sq + 1e-12, sigma_sq, epsilon);
    }
  }

  // Add harmonic contribution for the last particle (N)
  int start_idx_N = (N - 1) * n_spatial_dim + 1;
  int end_idx_N = N * n_spatial_dim;
  vector[n_spatial_dim] pos_N = x[start_idx_N:end_idx_N];
  harmonic_energy += dot_self(pos_N - x_com);

  // Finalize harmonic energy calculation
  harmonic_energy *= 0.5;

  // The target log-probability density is proportional to -Energy
  target += -(total_lj_energy + c_harmonic * harmonic_energy);

  // Add a standard normal prior to the positions for regularization
  x ~ std_normal();
}