#pragma once

template <int dim_>
struct TStateBase {
  using real = float;

  static constexpr int dim = dim_;
  int num_particles;
  int res[dim];

  real V_p = 10;   // TODO: variable vol
  real m_p = 100;  // TODO: variable m_p
  real E = 500;      // TODO: variable E
  real nu = 0.3;   // TODO: variable nu
  real mu = E / (2 * (1 + nu)), lambda = E * nu / ((1 + nu) * (1 - 2 * nu));

  TStateBase() {
    set(10, 100, 5, 0.3);
  }

  void set(real V_p, real m_p, real E, real nu) {
    this->V_p = V_p;
    this->m_p = m_p;
    this->E = E;
    this->nu = nu;
    this->mu = E / (2 * (1 + nu));
    this->lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  }

  real *x_storage;
  real *v_storage;
  real *F_storage;
  real *A_storage;
  real *P_storage;
  real *C_storage;
  real *grid_storage;
  real *grid_star_storage;

  real *grid_bc;

  real *grad_x_storage;
  real *grad_v_storage;
  real *grad_F_storage;
  real *grad_A_storage;
  real *grad_P_storage;
  real *grad_C_storage;
  real *grad_grid_storage;

  int num_cells;

  real gravity[dim];
  real dx, inv_dx, invD;
  real dt;

  std::vector<float> fetch_x();
  std::vector<float> fetch_grad_v();
  std::vector<float> fetch_grad_x();
  void set_initial_v(float *);
  void set_initial_F(float *F);
};

template<int dim>
void advance(TStateBase<dim> &state);
