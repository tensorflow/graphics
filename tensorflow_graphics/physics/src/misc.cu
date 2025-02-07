#include "kernels.h"
#include "linalg.h"
#include "state.cuh"
#include <cstdio>
#include <vector>

__global__ void saxpy(int n, real a, real *x, real *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

void saxpy_cuda(int N, real alpha, real *x, real *y) {
  real *d_x, *d_y;

  cudaMalloc(&d_x, N * sizeof(real));
  cudaMalloc(&d_y, N * sizeof(real));

  cudaMemcpy(d_x, x, N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(real), cudaMemcpyHostToDevice);

  saxpy<<<(N + 255) / 256, 256>>>(N, alpha, d_x, d_y);

  cudaMemcpy(y, d_y, N * sizeof(real), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
}

__global__ void test_svd(int n,
                         Matrix3 *A,
                         Matrix3 *U,
                         Matrix3 *sig,
                         Matrix3 *V) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    svd(A[id], U[id], sig[id], V[id]);
  }
}

// 3D only..
void test_svd_cuda(int n, real *A, real *U, real *sig, real *V) {
  Matrix3 *d_A, *d_U, *d_sig, *d_V;

  cudaMalloc(&d_A, sizeof(Matrix3) * (unsigned int)(n));
  cudaMemcpy(d_A, A, sizeof(Matrix3) * n, cudaMemcpyHostToDevice);

  cudaMalloc(&d_U, sizeof(Matrix3) * (unsigned int)(n));
  cudaMalloc(&d_sig, sizeof(Matrix3) * (unsigned int)(n));
  cudaMalloc(&d_V, sizeof(Matrix3) * (unsigned int)(n));

  test_svd<<<(n + 127) / 128, 128>>>(n, d_A, d_U, d_sig, d_V);

  std::vector<Matrix3> h_U(n), h_sig(n), h_V(n);
  cudaMemcpy(h_U.data(), d_U, sizeof(Matrix3) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_sig.data(), d_sig, sizeof(Matrix3) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_V.data(), d_V, sizeof(Matrix3) * n, cudaMemcpyDeviceToHost);

  // Taichi uses column-first storage
  for (int p = 0; p < n; p++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        U[p * 12 + 4 * i + j] = h_U[p][j][i];
        sig[p * 12 + 4 * i + j] = h_sig[p][j][i];
        V[p * 12 + 4 * i + j] = h_V[p][j][i];
      }
    }
  }
}

template <int dim>
__host__ std::vector<real> TStateBase<dim>::fetch_x() {
  std::vector<real> host_x(dim * num_particles);
  cudaMemcpy(host_x.data(), x_storage,
             sizeof(TVector<real, dim>) * num_particles,
             cudaMemcpyDeviceToHost);
  return host_x;
}

template <int dim>
__host__ std::vector<real> TStateBase<dim>::fetch_grad_v() {
  std::vector<real> host_grad_v(dim * num_particles);
  cudaMemcpy(host_grad_v.data(), grad_v_storage,
             sizeof(TVector<real, dim>) * num_particles,
             cudaMemcpyDeviceToHost);
  return host_grad_v;
}

template <int dim>
__host__ std::vector<real> TStateBase<dim>::fetch_grad_x() {
  std::vector<real> host_grad_x(dim * num_particles);
  cudaMemcpy(host_grad_x.data(), grad_x_storage,
             sizeof(TVector<real, dim>) * num_particles,
             cudaMemcpyDeviceToHost);
  return host_grad_x;
}

template <int dim>
void TStateBase<dim>::set_initial_v(float *v) {
  cudaMemcpy(v_storage, v, sizeof(real) * dim * num_particles,
             cudaMemcpyHostToDevice);
}

template <int dim>
void TStateBase<dim>::set_initial_F(float *F) {
  cudaMemcpy(F_storage, F, sizeof(real) * dim * dim * num_particles,
             cudaMemcpyHostToDevice);
}

template std::vector<float> TStateBase<2>::fetch_x();
template std::vector<float> TStateBase<3>::fetch_x();
template std::vector<float> TStateBase<2>::fetch_grad_x();
template std::vector<float> TStateBase<3>::fetch_grad_x();
template std::vector<float> TStateBase<2>::fetch_grad_v();
template std::vector<float> TStateBase<3>::fetch_grad_v();
template void TStateBase<2>::set_initial_F(float *);
template void TStateBase<3>::set_initial_F(float *);
template void TStateBase<2>::set_initial_v(float *);
template void TStateBase<3>::set_initial_v(float *);

template <int dim>
void set_mpm_bc(void *state_, float *bc) {
  auto state = reinterpret_cast<TState<dim> *>(state_);
  cudaMemcpy(state->grid_bc, bc,
             sizeof(TVector<real, dim + 1>) * state->num_cells,
             cudaMemcpyHostToDevice);
}

template void set_mpm_bc<2>(void *state_, float *bc);
template void set_mpm_bc<3>(void *state_, float *bc);

template <int dim>
void set_mpm_actuation(void *state_, float *act) {
  auto state = reinterpret_cast<TState<dim> *>(state_);
  cudaMemcpy(state->A_storage, act,
             sizeof(real) * dim * dim * state->num_particles,
             cudaMemcpyHostToDevice);
}

template void set_mpm_actuation<2>(void *state_, float *);
template void set_mpm_actuation<3>(void *state_, float *);
