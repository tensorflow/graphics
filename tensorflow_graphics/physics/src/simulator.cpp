#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/testing.h>
#include <taichi/io/optix.h>
#include <taichi/visual/gui.h>
#include <Partio.h>
#include <taichi/system/profiler.h>
#include <taichi/math/svd.h>
#include "config.h"
#include "kernels.h"
#include "state_base.h"

TC_NAMESPACE_BEGIN

void write_partio(std::vector<Vector3> positions,
                  const std::string &file_name) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  Partio::ParticleAttribute posH, vH, mH, typeH, normH, statH, boundH, distH,
      debugH, indexH, limitH, apicH;

  bool verbose = false;

  posH = parts->addAttribute("position", Partio::VECTOR, 3);
  // typeH = parts->addAttribute("type", Partio::INT, 1);
  // indexH = parts->addAttribute("index", Partio::INT, 1);
  // limitH = parts->addAttribute("limit", Partio::INT, 3);
  // vH = parts->addAttribute("v", Partio::VECTOR, 3);

  if (verbose) {
    mH = parts->addAttribute("m", Partio::VECTOR, 1);
    normH = parts->addAttribute("boundary_normal", Partio::VECTOR, 3);
    debugH = parts->addAttribute("debug", Partio::VECTOR, 3);
    statH = parts->addAttribute("states", Partio::INT, 1);
    distH = parts->addAttribute("boundary_distance", Partio::FLOAT, 1);
    boundH = parts->addAttribute("near_boundary", Partio::INT, 1);
    apicH = parts->addAttribute("apic_frobenius_norm", Partio::FLOAT, 1);
  }
  for (auto p : positions) {
    // const Particle *p = allocator.get_const(p_i);
    int idx = parts->addParticle();
    // Vector vel = p->get_velocity();
    // float32 *v_p = parts->dataWrite<float32>(vH, idx);
    // for (int k = 0; k < 3; k++)
    //  v_p[k] = vel[k];
    // int *type_p = parts->dataWrite<int>(typeH, idx);
    // int *index_p = parts->dataWrite<int>(indexH, idx);
    // int *limit_p = parts->dataWrite<int>(limitH, idx);
    float32 *p_p = parts->dataWrite<float32>(posH, idx);

    // Vector pos = p->pos;

    for (int k = 0; k < 3; k++)
      p_p[k] = 0.f;

    for (int k = 0; k < 3; k++)
      p_p[k] = p[k];
    // type_p[0] = int(p->is_rigid());
    // index_p[0] = p->id;
    // limit_p[0] = p->dt_limit;
    // limit_p[1] = p->stiffness_limit;
    // limit_p[2] = p->cfl_limit;
  }
  Partio::write(file_name.c_str(), *parts);
  parts->release();
}

auto gpu_mpm3d = []() {
  constexpr int dim = 3;
  int n = 16;
  int num_particles = n * n * n;
  std::vector<real> initial_positions;
  std::vector<real> initial_velocities;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        bool right = (i / (n / 2));
        // initial_positions.push_back(i * 0.025_f + 0.2123_f);
        initial_positions.push_back(i * 0.025_f + 0.2123_f + 0.2 * right);
        initial_velocities.push_back(1 - 1 * right);
        // initial_velocities.push_back(0.0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(j * 0.025_f + 0.4344_f);
        initial_velocities.push_back(0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(k * 0.025_f + 0.9854_f);
        initial_velocities.push_back(0);
      }
    }
  }
  std::vector<real> initial_F;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < num_particles; i++) {
        initial_F.push_back(real(a == b) * 1.1);
      }
    }
  }
  TC_P(initial_F.size());
  int num_steps = 80;
  std::vector<TStateBase<dim> *> states((uint32)num_steps + 1, nullptr);
  Vector3i res(20);
  // Differentiate gravity is not supported
  Vector3 gravity(0, 0, 0);
  for (int i = 0; i < num_steps + 1; i++) {
    initialize_mpm_state<3>(&res[0], num_particles, &gravity[0],
                            (void *&)states[i], 1.0_f / res[0], 5e-3_f,
                            initial_positions.data());
    std::fill(initial_positions.begin(), initial_positions.end(), 0);
    if (i == 0) {
      states[i]->set_initial_v(initial_velocities.data());
      states[i]->set_initial_F(initial_F.data());
    }
  }

  for (int i = 0; i < num_steps; i++) {
    TC_INFO("forward step {}", i);
    auto x = states[i]->fetch_x();
    OptiXScene scene;
    for (int p = 0; p < (int)initial_positions.size() / 3; p++) {
      OptiXParticle particle;
      auto scale = 5_f;
      particle.position_and_radius =
          Vector4(x[p] * scale, (x[p + num_particles] - 0.02f) * scale,
                  x[p + 2 * num_particles] * scale, 0.03);
      scene.particles.push_back(particle);
      if (p == 0)
        TC_P(particle.position_and_radius);
    }
    int interval = 3;
    if (i % interval == 0) {
      write_to_binary_file(scene, fmt::format("{:05d}.tcb", i / interval));
    }
    forward_mpm_state<dim>(states[i], states[i + 1]);
  }

  set_grad_loss(states[num_steps]);
  for (int i = num_steps - 1; i >= 0; i--) {
    TC_INFO("backward step {}", i);
    backward_mpm_state<dim>(states[i], states[i + 1]);
    auto grad_x = states[i]->fetch_grad_x();
    auto grad_v = states[i]->fetch_grad_v();
    Vector3f vgrad_v, vgrad_x;
    for (int j = 0; j < num_particles; j++) {
      for (int k = 0; k < 3; k++) {
        vgrad_v[k] += grad_v[k * num_particles + j];
        vgrad_x[k] += grad_x[k * num_particles + j];
      }
    }
    TC_P(vgrad_v);
    TC_P(vgrad_x);
  }
};

TC_REGISTER_TASK(gpu_mpm3d);

auto gpu_mpm3d_falling_leg = []() {
  // Actuation or falling test?
  bool actuation = true;
  constexpr int dim = 3;
  // The cube has size 2 * 2 * 2, with height 5m, falling time = 1s, g=-10
  int n = 20;
  real dx = 0.4;
  real sample_density = 0.1;
  Vector3 corner(20, 15 * (!actuation) + 20 * dx, 20);

  using Vector = Vector3;

  std::vector<Vector3> particle_positions;

  std::vector<int> actuator_indices;

  auto add_cube = [&](Vector corner, Vector size) {
    actuator_indices.clear();
    real d = sample_density;
    auto sizeI = (size / sample_density).template cast<int>();
    int padding = 3;
    int counter = 0;
    for (auto i : TRegion<3>(Vector3i(0), sizeI)) {
      counter++;
      particle_positions.push_back(Vector(
          corner[0] + d * i.i, corner[1] + d * i.j, corner[2] + d * i.k));
      if (padding <= i.i && padding <= i.j && padding <= i.k &&
          i.i + padding < sizeI[0] && i.j + padding < sizeI[1] &&
          i.k + padding < sizeI[2]) {
        actuator_indices.push_back(counter);
      }
    }
  };

  Vector chamber_size(2, 14.5, 2);
  add_cube(corner + chamber_size * Vector(1, 0, 0), chamber_size);
  add_cube(corner + chamber_size * Vector(0, 0, 1), chamber_size);
  add_cube(corner + chamber_size * Vector(1, 0, 1), chamber_size);
  add_cube(corner + chamber_size * Vector(2, 0, 1), chamber_size);
  add_cube(corner + chamber_size * Vector(1, 0, 2), chamber_size);

  int num_particles = particle_positions.size();
  std::vector<real> initial_positions;
  initial_positions.resize(particle_positions.size() * 3);
  for (int i = 0; i < particle_positions.size(); i++) {
    initial_positions[i] = particle_positions[i].x;
    initial_positions[i + particle_positions.size()] = particle_positions[i].y;
    initial_positions[i + 2 * particle_positions.size()] =
        particle_positions[i].z;
  }

  int num_frames = 200;
  Vector3i res(200, 120, 200);
  Array3D<Vector4f> bc(res);
  TC_WARN("Should run without APIC.");
  TC_WARN("Should use damping 2e-4 on grid.");
  for (int i = 0; i < res[0]; i++) {
    for (int j = 0; j < 22; j++) {
      for (int k = 0; k < res[2]; k++) {
        bc[i][j][k] = Vector4(0, 1, 0, -1);
      }
    }
  }
  // cm/s^2
  Vector3 gravity(0, -980.0, 0);
  TStateBase<dim> *state;
  TStateBase<dim> *state2;
  int substep = 160;
  real dt = 1.0_f / 120 / substep;
  initialize_mpm_state<3>(&res[0], num_particles, &gravity[0], (void *&)state,
                          dx, dt, initial_positions.data());
  set_mpm_bc<3>(state, &bc[0][0][0][0]);
  reinterpret_cast<TStateBase<3> *>(state)->set(10, 100, 50000000, 0.3);
  initialize_mpm_state<3>(&res[0], num_particles, &gravity[0], (void *&)state2,
                          dx, dt, initial_positions.data());
  set_mpm_bc<3>(state2, &bc[0][0][0][0]);
  reinterpret_cast<TStateBase<3> *>(state2)->set(10, 100, 50000000, 0.3);

  std::vector<real> A(num_particles * 9);

  for (int i = 0; i < num_frames; i++) {
    TC_INFO("forward step {}", i);
    auto x = state->fetch_x();
    auto fn = fmt::format("{:04d}.bgeo", i);
    TC_INFO(fn);
    std::vector<Vector3> parts;
    for (int p = 0; p < (int)initial_positions.size() / 3; p++) {
      auto pos = Vector3(x[p], x[p + num_particles], x[p + 2 * num_particles]);
      parts.push_back(pos);
    }
    write_partio(parts, fn);

    if (actuation) {
      for (int j = 0; j < actuator_indices.size(); j++) {
        real alpha = 500000;
        A[j + 0 * num_particles] = alpha * i;
        A[j + 4 * num_particles] = alpha * i;
        A[j + 8 * num_particles] = alpha * i;
      }
      set_mpm_actuation<dim>(state, A.data());
    }

    {
      TC_PROFILER("simulate one frame");
      for (int j = 0; j < substep; j++)
        forward_mpm_state<dim>(state, state);
    }
    // taichi::print_profile_info();
  }
};

TC_REGISTER_TASK(gpu_mpm3d_falling_leg);

auto gpu_mpm3d_falling_cube = []() {
  constexpr int dim = 3;
  // The cube has size 2 * 2 * 2, with height 5m, falling time = 1s, g=-10
  int n = 80;
  real dx = 0.2;
  real sample_density = 0.1;
  Vector3 corner(2, 5 + 2 * dx, 2);
  int num_particles = n * n * n;
  std::vector<real> initial_positions;
  std::vector<real> initial_velocities;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        // initial_positions.push_back(i * 0.025_f + 0.2123_f);
        initial_positions.push_back(i * sample_density + corner[0]);
        initial_velocities.push_back(0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(j * sample_density + corner[1]);
        initial_velocities.push_back(0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(k * sample_density + corner[2]);
        initial_velocities.push_back(0);
      }
    }
  }
  std::vector<real> initial_F;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < num_particles; i++) {
        initial_F.push_back(real(a == b) * 1.1);
      }
    }
  }
  int num_frames = 300;
  Vector3i res(100, 120, 100);
  Vector3 gravity(0, -10, 0);
  TStateBase<dim> *state;
  TStateBase<dim> *state2;
  int substep = 3;
  real dt = 1.0_f / 60 / substep;
  initialize_mpm_state<3>(&res[0], num_particles, &gravity[0], (void *&)state,
                          dx, dt, initial_positions.data());
  reinterpret_cast<TStateBase<3> *>(state)->set(10, 100, 5000, 0.3);
  initialize_mpm_state<3>(&res[0], num_particles, &gravity[0], (void *&)state2,
                          dx, dt, initial_positions.data());
  reinterpret_cast<TStateBase<3> *>(state2)->set(10, 100, 5000, 0.3);
  state->set_initial_v(initial_velocities.data());

  for (int i = 0; i < num_frames; i++) {
    TC_INFO("forward step {}", i);
    auto x = state->fetch_x();
    auto fn = fmt::format("{:04d}.bgeo", i);
    TC_INFO(fn);
    std::vector<Vector3> parts;
    for (int p = 0; p < (int)initial_positions.size() / 3; p++) {
      auto pos = Vector3(x[p], x[p + num_particles], x[p + 2 * num_particles]);
      parts.push_back(pos);
    }
    write_partio(parts, fn);

    {
      TC_PROFILER("simulate one frame");
      for (int j = 0; j < substep; j++)
        forward_mpm_state<dim>(state, state);
    }
    taichi::print_profile_info();
  }
  while (true) {
    TC_PROFILER("backward");
    for (int j = 0; j < substep; j++)
      backward_mpm_state<dim>(state2, state);
    taichi::print_profile_info();
  }
};

TC_REGISTER_TASK(gpu_mpm3d_falling_cube);

auto gpu_mpm2d_falling_cube = []() {
  constexpr int dim = 2;
  // The cube has size 2 * 2 * 2, with height 5m, falling time = 1s, g=-10
  int n = 80;
  real dx = 0.2;
  real sample_density = 0.1;
  Vector2 corner(2, 5 + 2 * dx);
  int num_particles = n * n;
  std::vector<real> initial_positions;
  std::vector<real> initial_velocities;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      initial_positions.push_back(i * sample_density + corner[0]);
      initial_velocities.push_back(0);
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      initial_positions.push_back(j * sample_density + corner[1]);
      initial_velocities.push_back(0);
    }
  }
  std::vector<real> initial_F;
  int num_frames = 3;
  Vector2i res(100, 120);
  Vector2 gravity(0, -10);
  TStateBase<dim> *state;
  TStateBase<dim> *state2;
  int substep = 3;
  real dt = 1.0_f / 60 / substep;
  initialize_mpm_state<2>(&res[0], num_particles, &gravity[0], (void *&)state,
                          dx, dt, initial_positions.data());
  reinterpret_cast<TStateBase<dim> *>(state)->set(10, 100, 5000, 0.3);
  initialize_mpm_state<2>(&res[0], num_particles, &gravity[0], (void *&)state2,
                          dx, dt, initial_positions.data());
  reinterpret_cast<TStateBase<dim> *>(state2)->set(10, 100, 5000, 0.3);
  state->set_initial_v(initial_velocities.data());

  for (int i = 0; i < num_frames; i++) {
    TC_INFO("forward step {}", i);
    auto x = state->fetch_x();
    auto fn = fmt::format("{:04d}.bgeo", i);
    TC_INFO(fn);
    std::vector<Vector3> parts;
    for (int p = 0; p < (int)initial_positions.size() / dim; p++) {
      auto pos = Vector3(x[p], x[p + num_particles], 0);
      parts.push_back(pos);
    }
    write_partio(parts, fn);

    {
      TC_PROFILER("simulate one frame");
      for (int j = 0; j < substep; j++)
        forward_mpm_state<dim>(state, state);
    }
    taichi::print_profile_info();
  }
  while (true) {
    TC_PROFILER("backward");
    for (int j = 0; j < substep; j++)
      backward_mpm_state<dim>(state2, state);
    taichi::print_profile_info();
  }
};

TC_REGISTER_TASK(gpu_mpm2d_falling_cube);

auto test_cuda = []() {
  int N = 10;
  std::vector<real> a(N), b(N);
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }
  saxpy_cuda(N, 2.0_f, a.data(), b.data());
  for (int i = 0; i < N; i++) {
    TC_ASSERT_EQUAL(b[i], i * 4.0_f, 1e-5_f);
  }
};

TC_REGISTER_TASK(test_cuda);

auto test_cuda_svd = []() {
  int N = 12800;
  using Matrix = Matrix3f;
  std::vector<Matrix> A, U, sig, V;
  A.resize(N);
  U.resize(N);
  sig.resize(N);
  V.resize(N);

  std::vector<real> A_flattened;
  for (int p = 0; p < N; p++) {
    auto matA = Matrix(1) + 0.5_f * Matrix::rand();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        A_flattened.push_back(matA(i, j));
      }
    }
    A[p] = matA;
  }

  test_svd_cuda(N, (real *)A_flattened.data(), (real *)U.data(),
                (real *)sig.data(), (real *)V.data());

  constexpr real tolerance = 3e-5_f32;
  for (int i = 0; i < N; i++) {
    auto matA = A[i];
    auto matU = U[i];
    auto matV = V[i];
    auto matSig = sig[i];

    TC_ASSERT_EQUAL(matSig, Matrix(matSig.diag()), tolerance);
    TC_ASSERT_EQUAL(Matrix(1), matU * transposed(matU), tolerance);
    TC_ASSERT_EQUAL(Matrix(1), matV * transposed(matV), tolerance);
    TC_ASSERT_EQUAL(matA, matU * matSig * transposed(matV), tolerance);

    /*
    polar_decomp(m, R, S);
    TC_CHECK_EQUAL(m, R * S, tolerance);
    TC_CHECK_EQUAL(Matrix(1), R * transposed(R), tolerance);
    TC_CHECK_EQUAL(S, transposed(S), tolerance);
    */
  }
};

TC_REGISTER_TASK(test_cuda_svd);

auto test_partio = []() {
  real dx = 0.01_f;
  for (int f = 0; f < 100; f++) {
    std::vector<Vector3> positions;
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        for (int k = 0; k < 10; k++) {
          positions.push_back(dx * Vector3(i + f, j, k));
        }
      }
    }
    auto fn = fmt::format("{:04d}.bgeo", f);
    TC_INFO(fn);
    write_partio(positions, fn);
  }
};

TC_REGISTER_TASK(test_partio);

auto write_partio_c = [](const std::vector<std::string> &parameters) {
  auto n = (int)std::atoi(parameters[0].c_str());
  float *pos_ = reinterpret_cast<float *>(std::atol(parameters[1].c_str()));
  auto fn = parameters[2];
  using namespace taichi;
  std::vector<Vector3> pos;
  for (int i = 0; i < n; i++) {
    auto p = Vector3(pos_[i], pos_[i + n], pos_[i + 2 * n]);
    pos.push_back(p);
  }
  taichi::write_partio(pos, fn);
};

TC_REGISTER_TASK(write_partio_c);

auto write_tcb_c = [](const std::vector<std::string> &parameters) {
  auto n = (int)std::atoi(parameters[0].c_str());
  float *pos_ = reinterpret_cast<float *>(std::atol(parameters[1].c_str()));
  auto fn = parameters[2];
  using namespace taichi;
  std::vector<Vector4> pos;
  for (int i = 0; i < n; i++) {
    auto p = Vector4(pos_[i], pos_[i + n], pos_[i + 2 * n], pos_[i + 3 * n]);
    pos.push_back(p);
  }
  write_to_binary_file(pos, fn);
};

TC_REGISTER_TASK(write_tcb_c);

TC_FORCE_INLINE void polar_decomp_simple(const TMatrix<real, 2> &m,
                                         TMatrix<real, 2> &R,
                                         TMatrix<real, 2> &S) {
  /*
  x = m[:, 0, 0, :] + m[:, 1, 1, :]
  y = m[:, 1, 0, :] - m[:, 0, 1, :]
  scale = 1.0 / tf.sqrt(x**2 + y**2)
  c = x * scale
  s = y * scale
  r = make_matrix2d(c, -s, s, c)
  return r, matmatmul(transpose(r), m)
  */

  auto x = m(0, 0) + m(1, 1);
  auto y = m(1, 0) - m(0, 1);
  auto scale = 1.0_f / std::sqrt(x * x + y * y);
  auto c = x * scale;
  auto s = y * scale;
  R = Matrix2(Vector2(c, s), Vector2(-s, c));
  S = transposed(R) * m;
}

TC_FORCE_INLINE TMatrix<real, 2> dR_from_dF(const TMatrix<real, 2> &F,
                                            const TMatrix<real, 2> &R,
                                            const TMatrix<real, 2> &S,
                                            const TMatrix<real, 2> &dF) {
  using Matrix = TMatrix<real, 2>;
  using Vector = TVector<real, 2>;
  // set W = R^T dR = [  0    x  ]
  //                  [  -x   0  ]
  //
  // R^T dF - dF^T R = WS + SW
  //
  // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
  //           [ -x[s11 + s22]  x(s21 - s12) ]
  // ----------------------------------------------------
  Matrix lhs = transposed(R) * dF - transposed(dF) * R;
  real x = lhs(0, 1) / (S(0, 0) + S(1, 1));
  Matrix W = Matrix(Vector(0, -x), Vector(x, 0));
  return R * W;
};

void Times_Rotated_dP_dF_FixedCorotated(real mu,
                                        real lambda,
                                        const Matrix2 &F,
                                        const TMatrix<real, 2> &dF,
                                        TMatrix<real, 2> &dP) {
  using Matrix = TMatrix<real, 2>;
  using Vector = TVector<real, 2>;

  const auto j = determinant(F);
  Matrix r, s;
  polar_decomp_simple(F, r, s);
  Matrix dR = dR_from_dF(F, r, s, dF);
  Matrix JFmT = Matrix(Vector(F(1, 1), -F(0, 1)), Vector(-F(1, 0), F(0, 0)));
  Matrix dJFmT =
      Matrix(Vector(dF(1, 1), -dF(0, 1)), Vector(-dF(1, 0), dF(0, 0)));
  dP = 2.0_f * mu * (dF - dR) +
       lambda * JFmT * (JFmT.elementwise_product(dF)).sum() +
       lambda * (j - 1) * dJFmT;
}

Matrix2 stress(real mu, real lambda, const Matrix2 &F) {
  Matrix2 r, s;
  polar_decomp(F, r, s);
  auto j = determinant(F);
  return 2.0_f * mu * (F - r) + lambda * j * (j - 1) * inverse(transposed(F));
}

auto test_2d_differential = []() {
  using Matrix = TMatrix<real, 2>;
  real mu = 13, lambda = 32;
  for (int r = 0; r < 10000; r++) {
    TC_INFO("{}", r);
    Matrix F = Matrix::rand();
    if (determinant(F) < 0.1_f) {
      // Assuming no negative
      continue;
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        Matrix dF, dP;
        dF(i, j) = 1;
        real s_delta = 1e-3;
        Matrix delta;
        delta(i, j) = s_delta;
        Times_Rotated_dP_dF_FixedCorotated(mu, lambda, F, dF, dP);
        auto dP_numerical =
            1.0_f / (s_delta * 2) *
            (stress(mu, lambda, F + delta) - stress(mu, lambda, F - delta));
        auto ratio = (dP - dP_numerical).frobenius_norm() / dP.frobenius_norm();
        TC_P(determinant(F));
        TC_P(dP);
        TC_P(dP_numerical);
        TC_ASSERT(ratio < 1e-3);
      }
    }
  }
};

TC_REGISTER_TASK(test_2d_differential);

auto view_txt = [](const std::vector<std::string> &parameters) {
  std::FILE *f = std::fopen(parameters[0].c_str(), "r");
  int window_size = 800;
  int scale = 10;
  GUI ui("particles", window_size, window_size);
  while (true) {
    char type[100];
    if (std::feof(f)) {
      break;
    }
    fscanf(f, "%s", type);
    printf("reading...\n");
    if (type[0] == 'p') {
      real x, y, r, g, b, a00, a01, a10, a11;
      fscanf(f, "%f %f %f %f %f  %f %f %f %f", &x, &y, &r, &g, &b, &a00, &a01,
             &a10, &a11);
      ui.get_canvas().img[Vector2i(x * scale, y * scale)] =
          Vector3f(r, g, a11 * 2);
    }
  }
  while (1)
    ui.update();
};

TC_REGISTER_TASK(view_txt);

auto convert_obj = [](const std::vector<std::string> &parameters) {
  for (auto fn : parameters) {
    std::FILE *f = std::fopen(fn.c_str(), "r");
    char s[1000], type[100];
    std::vector<Vector3> vec;
    while (std::fgets(s, 1000, f)) {
      real x, y, z;
      sscanf(s, "%s %f %f %f", type, &x, &y, &z);
      if (type[0] == 'v') {
        vec.push_back(Vector3(x, y, z));
      }
    }
    TC_P(vec.size());
    write_to_binary_file(vec, fn + ".tcb");
  }
};

TC_REGISTER_TASK(convert_obj);

auto fuse_frames = [](const std::vector<std::string> &parameters) {
  TC_ASSERT(parameters.size() == 4);
  int iterations = std::atoi(parameters[0].c_str());
  int iteration_interval = std::atoi(parameters[1].c_str());
  int iteration_offset = 0;
  if (iteration_interval == 0) {
    iteration_offset = iterations;
    iterations = 1;
  }
  int frames = std::atoi(parameters[2].c_str());
  int frame_interval = std::atoi(parameters[3].c_str());
  for (int i = 0; i < frames; i++) {
    auto frame_fn = fmt::format("frame{:05d}.txt", i * frame_interval);
    std::vector<Vector3> vec;
    for (int j = 0; j < iterations; j++) {
      auto iteration_fn = fmt::format(
          "iteration{:04d}", j * iteration_interval + iteration_offset);
      std::FILE *f = std::fopen((iteration_fn + "/" + frame_fn).c_str(), "r");
      char s[1000], type[100];
      while (std::fgets(s, 1000, f)) {
        real x, y, a, _;
        sscanf(s, "%s %f %f %f %f %f %f %f %f %f", type, &x, &y, &_, &_, &_, &_,
               &_, &_, &a);
        if (type[0] == 'p') {
          x = x / 40;
          y = y / 40;
          vec.push_back(Vector3(x, y, -j * 0.09, a));
        }
      }
      fclose(f);
    }
    TC_P(vec.size());
    std::string prefix;
    if (iteration_interval == 0) {
      prefix = fmt::format("iteration{:04d}", iteration_offset);
    }
    write_to_binary_file(vec, prefix + frame_fn + ".tcb");
  }
};

TC_REGISTER_TASK(fuse_frames);

auto fuse_frames_ppo = [](const std::vector<std::string> &parameters) {
  TC_ASSERT(parameters.size() == 4);
  int iterations = std::atoi(parameters[0].c_str());
  int iteration_interval = std::atoi(parameters[1].c_str());
  int iteration_offset = 0;
  if (iteration_interval == 0) {
    iteration_offset = iterations;
    iterations = 1;
  }

  int frames = std::atoi(parameters[2].c_str());
  int frame_interval = std::atoi(parameters[3].c_str());

  /*
  auto fn = [&](int iteration, int frame) {
    auto frame_fn = fmt::format("frame{:05d}.txt", frame * frame_interval);
    auto iteration_fn =
        fmt::format("it{:04d}", iteration * iteration_interval + iteration_offset);
    return iteration_fn + "/" + frame_fn;
  };
  */
  auto fn = [&](int iteration, int frame) {
    // PPO
    auto frame_fn = fmt::format("it{:04d}/frame00001.txt", frame * frame_interval);
    auto iteration_fn =
        fmt::format("ep{:04d}", (iteration * iteration_interval + iteration_offset + 1) * 10);
    return iteration_fn + "/" + frame_fn;
  };

  for (int i = 0; i < frames; i++) {
    std::vector<Vector4> vec;
    for (int j = 0; j < iterations; j++) {
      auto fname = fn(j, i).c_str();
      // TC_P(fname);
      std::FILE *f = std::fopen(fname, "r");
      TC_ASSERT(f != nullptr);
      char s[1000], type[100];
      real c = -20.7_f;
      while (std::fgets(s, 1000, f)) {
        real x, y, a, _;
        sscanf(s, "%s %f %f %f %f %f %f %f %f %f", type, &x, &y, &_, &_, &_, &_,
               &_, &_, &a);
        Vector4 position_offset = Vector4(j * 0.1, 0, -j * 0.19, 0);
        if (type[0] == 'p') {
          x = x / 40;
          y = y / 40;
          vec.push_back(Vector4(x, y, 0, a) + position_offset);
        } else if (type[0] == 'v') {
          x /= 900;
          y /= 900;
          for (auto o : TRegion<3>(Vector3i(-2), Vector3i(3))) {
            real delta = 0.005;
            vec.push_back(
                Vector4(x + delta * o.i, y + delta * o.j, delta * o.k, c) +
                position_offset);
          }
          c = 20.7;
        }
      }
      fclose(f);
    }
    TC_P(vec.size());
    std::string prefix;
    if (iteration_interval == 0) {
      prefix = fmt::format("iteration{:04d}", iteration_offset);
    }
    write_to_binary_file(vec, prefix + fmt::format("frame{:04d}.tcb", i));
  }
};

TC_REGISTER_TASK(fuse_frames_ppo);

TC_NAMESPACE_END
