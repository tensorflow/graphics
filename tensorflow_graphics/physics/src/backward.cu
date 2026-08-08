#include "kernels.h"
#include "linalg.h"
#include "state.cuh"
#include <cstdio>
#include <vector>

template <int dim>
__global__ void P2G_backward(TState<dim> state, TState<dim> next_state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation
  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  auto x = state.get_x(part_id);
  auto v = state.get_v(part_id);
  auto F = state.get_F(part_id);
  auto C = state.get_C(part_id);

  auto grad_x_next = next_state.get_grad_x(part_id);
  auto grad_v_next = next_state.get_grad_v(part_id);
  auto grad_F_next = next_state.get_grad_F(part_id);
  auto grad_C_next = next_state.get_grad_C(part_id);

  // (A) v_p^n+1, accumulate
  grad_v_next = grad_v_next + state.dt * grad_x_next;

  // (B) C_p^n+1, accumulate
  for (int alpha = 0; alpha < dim; alpha++) {
    for (int beta = 0; beta < dim; beta++) {
      for (int gamma = 0; gamma < dim; gamma++) {
        grad_C_next[alpha][beta] +=
            state.dt * grad_F_next[alpha][gamma] * F[beta][gamma];
      }
    }
  }

  // Accumulate to grad_v and grad_C
  // next_state.set_grad_v(part_id, grad_v_next);
  // next_state.set_grad_C(part_id, grad_C_next);

  TransferCommon<dim, true> tc(state, x);

  for (int i = 0; i < kernel_volume<dim>(); i++) {
    real N = tc.w(i);
    auto dpos = tc.dpos(i);

    // (C) v_i^n
    real grad_v_i[dim];
    for (int alpha = 0; alpha < dim; alpha++) {
      grad_v_i[alpha] = grad_v_next[alpha] * N;
      for (int beta = 0; beta < dim; beta++) {
        grad_v_i[alpha] +=
            state.invD * N * grad_C_next[alpha][beta] * dpos[beta];
      }
    }
    auto grad_n =
        state.grad_grid_node(tc.base_coord + offset_from_scalar<dim>(i));
    for (int d = 0; d < dim; d++) {
      // printf("grad_v_i %d %f\n", d, grad_v_i[d]);
      atomicAdd(&grad_n[d], grad_v_i[d]);
    }
  }
}

TC_FORCE_INLINE __device__ real H(real x) {
  return x >= 0 ? 1 : 0;
}

template <int dim>
__global__ void grid_backward(TState<dim> state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation

  using Vector = typename TState<dim>::Vector;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < state.num_cells) {
    auto node = state.grid_node(id);
    auto grad_node = state.grad_grid_node(id);
    if (node[dim] > 0) {
      // (D)
      // Convert grad_v to grad_p
      // grad_p = grad_v / m
      auto m = node[dim];
      real inv_m = 1.0f / m;

      auto grad_v_i = Vector(grad_node);
      auto v_i = Vector(state.grid_star_node(id));
      auto v_i_with_g = v_i;
      for (int i = 0; i < dim; i++) {
        v_i_with_g[i] += state.gravity[i] * state.dt;
      }
      auto v_i_star = Vector(state.grid_node(id));

      auto bc = state.grid_node_bc(id);
      auto normal = Vector(bc);
      auto lin = v_i_with_g.dot(normal);

      real coeff = bc[dim];

      if (coeff == -1) {
        // sticky
        grad_v_i = Vector(0.0f);
      } else if (normal.length2() > 0) {
        auto vit = v_i_with_g - lin * normal;
        auto lit = sqrt(vit.length2() + 1e-7);
        auto vithat = (1.0f / lit) * vit;
        auto R = lit + coeff * min(lin, 0.0f);
        auto litstar = max(R, 0.0f);
        auto vistar = litstar * vithat + max(lin, 0.0f) * normal;

        auto r = vistar - v_i_star;
        for (int i = 0; i < dim; i++) {
          if (fabs(r[i]) > 1e-6)
            printf("mismatch r %f\n", r[i]);
        }
        auto grad_v_i_star = grad_v_i;

        auto grad_litstar = 0.0f;
        for (int i = 0; i < dim; i++) {
          grad_litstar += grad_v_i_star[i] * vithat[i];
        }
        Vector grad_vithat = litstar * grad_v_i_star;

        auto grad_lit = grad_litstar * H(R);
        for (int i = 0; i < dim; i++) {
          grad_lit += -1 / (lit * lit) * vit[i] * grad_vithat[i];
        }
        auto grad_vit = (1 / lit) * (grad_lit * vit + grad_vithat);
        auto grad_lin = grad_litstar * H(R) * coeff * H(-lin);

        for (int i = 0; i < dim; i++) {
          grad_lin -= grad_vit[i] * normal[i];
          grad_lin += H(lin) * normal[i] * grad_v_i_star[i];
        }
        auto new_grad_v_i = grad_lin * normal + grad_vit;
        grad_v_i = new_grad_v_i;
      }

      auto grad_p = inv_m * grad_v_i;
      // (E)
      real grad_m = 0;
      for (int alpha = 0; alpha < dim; alpha++) {
        grad_m -= inv_m * v_i[alpha] * grad_v_i[alpha];
        grad_node[alpha] = grad_p[alpha];
      }
      grad_node[dim] = grad_m;
    }
  }
}

// (F), (G), (H), (I), (J)
template <int dim>
__global__ void G2P_backward(TState<dim> state, TState<dim> next_state) {
  // Scatter particle gradients to grid nodes
  // P2G part of back-propagation
  int part_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_id >= state.num_particles) {
    return;
  }

  auto x = state.get_x(part_id);
  auto v = state.get_v(part_id);
  auto F = state.get_F(part_id);
  auto C = state.get_C(part_id);
  auto P = state.get_P(part_id);
  auto A = state.get_A(part_id);

  auto grad_F_next = next_state.get_grad_F(part_id);
  auto grad_C_next = next_state.get_grad_C(part_id);
  auto grad_P_next = next_state.get_grad_P(part_id);
  auto grad_v_next = next_state.get_grad_v(part_id);
  auto grad_x_next = next_state.get_grad_x(part_id);

  auto C_next = next_state.get_C(part_id);
  // (A) v_p^n+1, accumulate
  grad_v_next = grad_v_next + state.dt * grad_x_next;

  // (B) C_p^n+1, accumulate
  for (int alpha = 0; alpha < dim; alpha++) {
    for (int beta = 0; beta < dim; beta++) {
      for (int gamma = 0; gamma < dim; gamma++) {
        grad_C_next[alpha][beta] +=
            state.dt * grad_F_next[alpha][gamma] * F[beta][gamma];
      }
    }
  }

  TMatrix<real, dim> grad_P, grad_F, grad_C;

  TransferCommon<dim, true> tc(state, x);
  {
    /*
    real dx = 1e-4f;
    TransferCommon<true> tc2(state, x + Vector(0, 0, dx));
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          auto d = tc.dw(i, j, k);
          printf("%f %f\n", d[2], (tc2.w(i, j, k) - tc.w(i, j, k))/ dx);
        }
      }
    }
    */
  }

  TVector<real, dim> grad_v;
  real grad_P_scale = state.dt * state.invD * state.V_p;

  // (G) Compute grad_P
  for (int i = 0; i < kernel_volume<dim>(); i++) {
    real N = tc.w(i);
    auto dpos = tc.dpos(i);
    auto grad_p = state.get_grad_grid_velocity(tc.base_coord +
                                               offset_from_scalar<dim>(i));
    auto grad_N = tc.dw(i);
    for (int alpha = 0; alpha < dim; alpha++) {
      for (int beta = 0; beta < dim; beta++) {
        // (G) P_p^n
        for (int gamma = 0; gamma < dim; gamma++) {
          grad_P[alpha][beta] +=
              -N * grad_P_scale * grad_p[alpha] * F[gamma][beta] * dpos[gamma];
        }
        // (I) C_p^n
        if (mpm_enalbe_apic)
          grad_C[alpha][beta] += N * grad_p[alpha] * state.m_p * dpos[beta];
      }
    }
  }

  // (H) term 2
  if (mpm_enalbe_force) {
    Times_Rotated_dP_dF_FixedCorotated(state.mu, state.lambda, F, grad_P,
                                       grad_F);
    /*
    TMatrix<real, dim> grad_F2;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        auto inc = F, dec = F;
        real delta = 1e-4f;
        inc[i][j] += delta;
        dec[i][j] -= delta;
        auto diff = (1 / (2 * delta)) * (PK1(state.mu, state.lambda, inc) -
                                         PK1(state.mu, state.lambda, dec));
        grad_F2 = grad_F2 + grad_P[i][j] * diff;
      }
    }
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        printf("%d %d:  %f %f\n", i, j, grad_F2[i][j] * 1e8,
               grad_F[i][j] * 1e8);
      }
    }

    grad_F = grad_F2;
    */
  }

  for (int alpha = 0; alpha < dim; alpha++) {
    for (int beta = 0; beta < dim; beta++) {
      // (H) term 1
      for (int gamma = 0; gamma < dim; gamma++) {
        grad_F[alpha][beta] +=
            grad_F_next[gamma][beta] *
            (real(gamma == alpha) + state.dt * C_next[gamma][alpha]) +
            grad_P[alpha][gamma] * A[beta][gamma];
      }
    }
  }

  typename TState<dim>::Matrix grad_A;
  for (int alpha = 0; alpha < dim; alpha++) {
    for (int beta = 0; beta < dim; beta++) {
      for (int gamma = 0; gamma < dim; gamma++) {
        grad_A[alpha][beta] +=
            grad_P[gamma][beta] * F[gamma][alpha];
      }
    }
  }
  state.set_grad_A(part_id, grad_A);

  // (J) term 1
  auto grad_x = next_state.get_grad_x(part_id);
  // printf("grad_x %f\n", grad_x[0]);
  auto G = (mpm_enalbe_force) * -state.invD * state.dt * state.V_p * P *
           transposed(F);
  if (mpm_enalbe_apic) {
    G = G + state.m_p * C;
  }

  for (int i = 0; i < kernel_volume<dim>(); i++) {
    real N = tc.w(i);
    auto dpos = tc.dpos(i);
    auto grid_coord = tc.base_coord + offset_from_scalar<dim>(i);
    auto grad_p = state.get_grad_grid_velocity(grid_coord);

    for (int d = 0; d < dim; d++) {
      // printf("grad p[%d] %.10f\n", d, grad_p[d]);
    }

    auto grad_N = tc.dw(i);
    auto n = state.grid_node(grid_coord);
    auto mi = state.get_grid_mass(grid_coord);
    // printf(" m m %f %f\n", mi, n[dim]);
    auto vi = state.get_grid_velocity(grid_coord);
    auto grad_mi = state.grad_grid_node(grid_coord)[dim];

    // printf("%.10f\n", grad_p[0]);
    // printf("%.10f\n", grad_p[1]);
    // printf("%.10f\n", grad_p[2]);
    // printf("\n");
    for (int alpha = 0; alpha < dim; alpha++) {
      // (F) v_p^n
      grad_v[alpha] += N * state.m_p * grad_p[alpha];

      // (J) term 5
      grad_x[alpha] += grad_N[alpha] * grad_mi * state.m_p;

      for (int beta = 0; beta < dim; beta++) {
        for (int gamma = 0; gamma < dim; gamma++) {
          // (H), term 3
          grad_F[alpha][beta] +=
              -N * grad_p[gamma] * grad_P_scale * P[gamma][beta] * dpos[alpha];
        }

        // (J), term 2
        grad_x[alpha] += grad_v_next[beta] * grad_N[alpha] * vi[beta];
        // (J), term 3
        auto tmp = -grad_C_next[beta][alpha] * N * vi[beta];
        for (int gamma = 0; gamma < dim; gamma++) {
          tmp +=
              grad_C_next[beta][gamma] * grad_N[alpha] * vi[beta] * dpos[gamma];
        }
        grad_x[alpha] += state.invD * tmp;
        // auto tmp = grad_N[alpha] * vi[beta] * dpos[alpha] - N * vi[beta];
        // grad_x[alpha] += state.invD * grad_C_next[beta][alpha] * tmp;
        // (J), term 4
        grad_x[alpha] +=
            grad_p[beta] *
            (grad_N[alpha] * (state.m_p * v[beta] + (G * dpos)[beta]) -
             N * G[beta][alpha]);
      }
    }
  }
  state.set_grad_x(part_id, grad_x);
  /*
  for (int i = 0; i < dim; i++) {
    printf("v %d %f %f\n", i, grad_v[i], grad_x[i]);
  }
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      printf("m %d %d %f %f %f %f\n", i, j, grad_F[i][j], grad_C[i][j], F[i][j],
  grad_P[i][j]);
    }
  }
  */
  state.set_grad_v(part_id, grad_v);
  if (mpm_enalbe_force)
    state.set_grad_F(part_id, grad_F);
  state.set_grad_C(part_id, grad_C);
}

__device__ real rand_real(int i) {
  real t = sinf(i) * 100.0;
  return t - floor(t);
}

__global__ void check2d(int k_) {
  constexpr int dim = 2;
  int k = k_;
  auto rand = [&]() { return rand_real(k++); };
  using Vector = TVector<real, 2>;
  auto grad_v_i = Vector(rand(), rand());
  auto v_i = Vector(rand() * 2 - 1, rand() * 2 - 1);
  // auto v_i = Vector(-0.5, 0.0);

  auto angle = rand() * 2 * 3.14f;
  auto normal = Vector(sinf(angle), cosf(angle));
  // auto normal = Vector(1, 0);
  auto coeff = rand();
  // auto coeff = 0;

  auto forward = [&](Vector v_i) {
    auto lin = v_i.dot(normal);
    auto vit = v_i - lin * normal;
    auto lit = sqrt(vit.length2() + 1e-7);
    auto vithat = (1.0f / lit) * vit;
    auto R = lit + coeff * min(lin, 0.0f);
    auto litstar = max(R, 0.0f);
    auto vistar = litstar * vithat + max(lin, 0.0f) * normal;
    return vistar.dot(grad_v_i);
  };

  auto lin = v_i.dot(normal);
  auto vit = v_i - lin * normal;
  auto lit = sqrt(vit.length2() + 1e-7);
  auto vithat = (1.0f / lit) * vit;
  auto R = lit + coeff * min(lin, 0.0f);
  auto litstar = max(R, 0.0f);
  auto vistar = litstar * vithat + max(lin, 0.0f) * normal;

  auto grad_v_i_star = grad_v_i;

  auto grad_litstar = 0.0f;
  for (int i = 0; i < dim; i++) {
    grad_litstar += grad_v_i_star[i] * vithat[i];
  }
  Vector grad_vithat = litstar * grad_v_i_star;

  auto grad_lit = grad_litstar * H(R);
  for (int i = 0; i < dim; i++) {
    grad_lit += -1 / (lit * lit) * vit[i] * grad_vithat[i];
  }
  auto grad_vit = (1 / lit) * (grad_lit * vit + grad_vithat);
  auto grad_lin = grad_litstar * H(R) * coeff * H(-lin);
  /*
  printf("lit %f\n", lit);
  for (int i = 0; i < dim; i++) {
    printf("gradlitstar %f\n", grad_litstar);
  }
  printf("gradlin %f\n", grad_lin);
  */

  for (int i = 0; i < dim; i++) {
    // printf("normal [%d] %f\n", i, normal[i]);
    grad_lin -= grad_vit[i] * normal[i];
    grad_lin += H(lin) * normal[i] * grad_v_i_star[i];
  }
  auto new_grad_v_i = grad_lin * normal + grad_vit;

  real dx = 1e-4f;
  for (int d = 0; d < dim; d++) {
    Vector delta;
    delta[d] = dx;
    real f0 = forward(v_i + delta);
    real f1 = forward(v_i - delta);
    real grad = (f0 - f1) / (2 * dx);
    // printf("f0, 1 = %f %f\n", f0, f1);
    if (fabs(grad - new_grad_v_i[d]) > 1e-3f) {
      printf("errr %d   %f %f\n", d, grad, new_grad_v_i[d]);
    } else {
      // printf("pass %d   %f %f\n", d, grad, new_grad_v_i[d]);
    }
  }
}

template <int dim>
void backward(TState<dim> &state, TState<dim> &next) {
  state.clear_gradients();
  int num_blocks =
      (state.num_particles + particle_block_dim - 1) / particle_block_dim;
  int num_blocks_grid = state.grid_size();
  P2G_backward<dim><<<num_blocks, particle_block_dim>>>(state, next);
  auto err = cudaThreadSynchronize();
  grid_backward<dim>
      <<<state.num_cells / grid_block_dim + 1, grid_block_dim>>>(state);
  G2P_backward<dim><<<num_blocks, particle_block_dim>>>(state, next);
  if (err) {
    printf("Launch: %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void MPMGradKernelLauncher(int dim,
                           int *res,
                           int num_particles,
                           real dx,
                           real dt,
                           real E,
                           real nu,
                           real m_p,
                           real V_p,
                           real *gravity,
                           const real *inx,
                           const real *inv,
                           const real *inF,
                           const real *inC,
                           const real *inA,
                           const real *ingrid,
                           const real *outx,
                           const real *outv,
                           const real *outF,
                           const real *outC,
                           const real *outP,
                           const real *outgrid,
                           const real *outgrid_star,
                           real *grad_inx,
                           real *grad_inv,
                           real *grad_inF,
                           real *grad_inC,
                           real *grad_inA,
                           real *grad_ingrid,
                           const real *grad_outx,
                           const real *grad_outv,
                           const real *grad_outF,
                           const real *grad_outC,
                           const real *grad_outP,
                           const real *grad_outgrid,
                           const real *grad_outgrid_star) {
  if (dim == 2) {
    /*
    for (int i = 0; i < 10000; i++) {
      check2d<<<1, 1>>>(i * 100);
    }
    exit(-1);
    */
    constexpr int dim = 2;
    auto current = new TState<dim>(
        res, num_particles, dx, dt, gravity, (real *)inx, (real *)inv,
        (real *)inF, (real *)inC, (real *)outP, (real *)inA, (real *)outgrid,
        grad_inx, grad_inv, grad_inF, grad_inC, grad_inA, (real *)grad_outP,
        (real *)grad_outgrid);
    current->grid_bc = const_cast<real *>(ingrid);
    current->grid_star_storage = const_cast<real *>(outgrid_star);
    current->set(V_p, m_p, E, nu);
    auto next = new TState<dim>(
        res, num_particles, dx, dt, gravity, (real *)outx, (real *)outv,
        (real *)outF, (real *)outC, nullptr, nullptr, nullptr,
        (real *)grad_outx, (real *)grad_outv, (real *)grad_outF,
        (real *)grad_outC, nullptr, nullptr, nullptr);
    next->set(V_p, m_p, E, nu);
    backward<dim>(*current, *next);
  } else {
    constexpr int dim = 3;
    auto current = new TState<dim>(
        res, num_particles, dx, dt, gravity, (real *)inx, (real *)inv,
        (real *)inF, (real *)inC, (real *)outP, (real *)inA, (real *)outgrid,
        grad_inx, grad_inv, grad_inF, grad_inC, grad_inA, (real *)grad_outP,
        (real *)grad_outgrid);
    current->grid_bc = const_cast<real *>(ingrid);
    current->grid_star_storage = const_cast<real *>(outgrid_star);
    current->set(V_p, m_p, E, nu);
    auto next = new TState<dim>(
        res, num_particles, dx, dt, gravity, (real *)outx, (real *)outv,
        (real *)outF, (real *)outC, nullptr, nullptr, nullptr,
        (real *)grad_outx, (real *)grad_outv, (real *)grad_outF,
        (real *)grad_outC, nullptr, nullptr, nullptr);
    next->set(V_p, m_p, E, nu);
    backward<dim>(*current, *next);
  }
}

template <int dim>
void backward_mpm_state(void *state_, void *next_state_) {
  TState<dim> *state = reinterpret_cast<TState<dim> *>(state_);
  TState<dim> *next_state = reinterpret_cast<TState<dim> *>(next_state_);
  backward<dim>(*state, *next_state);
}

template void backward_mpm_state<2>(void *state_, void *next_state_);
template void backward_mpm_state<3>(void *state_, void *next_state_);

void set_grad_loss(void *state_) {
  constexpr int dim = 3;
  TState<3> *state = reinterpret_cast<TState<3> *>(state_);
  state->clear_gradients();
  int num_particles = state->num_particles;
  std::vector<float> grad_x_host(num_particles * dim);
  for (int i = 0; i < num_particles; i++) {
    grad_x_host[i] = 1.0f / num_particles;
  }
  cudaMemcpy(state->grad_x_storage, grad_x_host.data(),
             sizeof(real) * dim * num_particles, cudaMemcpyHostToDevice);
}
