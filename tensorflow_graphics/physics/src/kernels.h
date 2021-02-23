#pragma once
#include <vector>

void saxpy_cuda(int n, float alpha, float *x, float *y);
void test_svd_cuda(int n, float *, float *, float *, float *);
template <int dim>
void initialize_mpm_state(int *,
                          int,
                          float *,
                          void *&,
                          float dx,
                          float dt,
                          float *initial_positions
                          );

template <int dim>
void forward_mpm_state(void *, void *);

template <int dim>
void backward_mpm_state(void *, void *);

void set_grad_loss(void *);

template <int dim>
void set_mpm_bc(void *state_, float *bc);

template <int dim>
void set_mpm_actuation(void *state_, float *act);
