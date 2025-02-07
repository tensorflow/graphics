import unittest
import time
from simulation import Simulation
from time_integration import UpdatedSimulationState
import tensorflow as tf
import numpy as np
from vector_math import *

sess = tf.compat.v1.Session()


class TestSimulator2D(unittest.TestCase):

  def assertAlmostEqualFloat32(self, a, b, relative_tol=1e-5, clip=1e-3):
    if abs(a - b) > relative_tol * max(max(abs(a), abs(b)), clip):
      self.assertEqual(a, b)

  def motion_test(self,
                  gravity=(0, -10),
                  initial_velocity=(0, 0),
                  batch_size=1,
                  dx=1.0,
                  num_steps=10):
    # Zero gravity, 1-batched, translating block
    num_particles = 100
    sim = Simulation(
        sess=sess,
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        batch_size=batch_size)
    initial = sim.initial_state
    next_state = UpdatedSimulationState(sim, initial)
    position = np.zeros(shape=(batch_size, 2, num_particles))
    velocity = np.zeros(shape=(batch_size, 2, num_particles))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, :, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
          velocity[b, :, i * 10 + j] = initial_velocity
    input_state = sim.get_initial_state(position=position, velocity=velocity)

    def center_of_mass():
      return np.mean(input_state[0][:, 0, :]), np.mean(input_state[0][:, 1, :])

    x, y = 15.0 * dx, 15.0 * dx
    vx, vy = initial_velocity

    self.assertAlmostEqual(center_of_mass()[0], x)
    self.assertAlmostEqual(center_of_mass()[1], y)
    for i in range(num_steps):
      input_state = sess.run(
          next_state.to_tuple(),
          feed_dict={sim.initial_state_place_holder(): input_state})

      # This will work if we use Verlet
      # self.assertAlmostEqual(center_of_mass()[1], 15.0 - t * t * 0.5 * g)

      # Symplectic Euler version
      vx += sim.dt * gravity[0]
      x += sim.dt * vx
      vy += sim.dt * gravity[1]
      y += sim.dt * vy
      self.assertAlmostEqualFloat32(center_of_mass()[0], x)
      self.assertAlmostEqualFloat32(center_of_mass()[1], y)

  def test_translation_x(self):
    self.motion_test(initial_velocity=(1, 0))

  def test_translation_x_batched(self):
    self.motion_test(initial_velocity=(1, 0), batch_size=1)

  def test_translation_y(self):
    self.motion_test(initial_velocity=(0, 1))

  def test_falling_translation(self):
    self.motion_test(initial_velocity=(2, -1), gravity=(-4, 6))

  def test_falling_translation_dx(self):
    self.motion_test(initial_velocity=(2, -1), gravity=(-4, 6), dx=0.05)
    self.motion_test(
        initial_velocity=(0.02, -0.01), gravity=(-0.04, 0.06), dx=0.1)
    self.motion_test(initial_velocity=(2, -1), gravity=(-4, 6), dx=10)

  def test_free_fall(self):
    self.motion_test(gravity=(0, -10))

  '''
  def test_recursive_placeholder(self):
    a = tf.placeholder(dtype=tf_precision)
    b = tf.placeholder(dtype=tf_precision)
    self.assertAlmostEqual(sess.run(a + b, feed_dict={(a, b): [1, 2]}), 3)
    # The following will not work
    # print(sess.run(a + b, feed_dict={{'a':a, 'b':b}: {'a':1, 'b':2}}))
  '''

  def test_bouncing_cube(self):
    #gravity = (0, -10)
    gravity = (0, -10)
    batch_size = 1
    dx = 0.03
    num_particles = 100
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=1e-3,
        E=100,
        nu=0.45,
        batch_size=batch_size,
        sess=sess)
    position = np.zeros(shape=(batch_size, 2, num_particles))
    initial_velocity = tf.compat.v1.placeholder(shape=(2,), dtype=tf_precision)
    velocity = tf.broadcast_to(
        initial_velocity[None, :, None], shape=(batch_size, 2, num_particles))

    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, :, i * 10 + j] = (((i + b * 3) * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
    input_state = sim.get_initial_state(
        position=position,
        velocity=velocity,
        )

    memo = sim.run(
        num_steps=1000,
        initial_state=input_state,
        initial_feed_dict={initial_velocity: [1, -2]})
    sim.visualize(memo, interval=5)

  def test_bouncing_cube_benchmark(self):
    gravity = (0, -10)
    batch_size = 1
    dx = 0.2
    sample_density = 0.1
    N = 80
    num_particles = N * N
    sim = Simulation(
      grid_res=(100, 120),
      dx=dx,
      num_particles=num_particles,
      gravity=gravity,
      dt=1 / 60 / 3,
      batch_size=batch_size,
      sess=sess)
    position = np.zeros(shape=(batch_size, 2, num_particles))
    poissons_ratio = np.ones(shape=(batch_size, 1, num_particles)) * 0.3
    velocity_ph = tf.compat.v1.placeholder(shape=(2,), dtype=tf_precision)
    for b in range(batch_size):
      for i in range(N):
        for j in range(N):
          position[b, :, i * N + j] = (i * sample_density + 2, j * sample_density + 5 + 2 * dx)
          
    
    loss = tf.reduce_mean(input_tensor=sim.initial_state.center_of_mass()[:, 0])

    velocity_ph = tf.compat.v1.placeholder(shape=(2,), dtype=tf_precision)
    velocity = velocity_ph[None, :, None] + tf.zeros(
      shape=[batch_size, 2, num_particles], dtype=tf_precision)
    input_state = sim.get_initial_state(
      position=position,
      velocity=velocity,
      poissons_ratio=poissons_ratio,
      youngs_modulus=1000)
    
    sim.set_initial_state(initial_state=input_state)
    import time
    memo = sim.run(
      num_steps=100,
      initial_state=input_state,
      initial_feed_dict={velocity_ph: [0, 0]})
    sym = sim.gradients_sym(loss=loss, variables=[velocity_ph])
    while True:
      t = time.time()
      grad = sim.eval_gradients(sym, memo)
      print((time.time() - t) / 100 * 3)
    sim.visualize(memo, interval=5)

  def test_rotating_cube(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    num_particles = 100
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=1e-4,
        E=1,
        sess=sess)
    position = np.zeros(shape=(batch_size, 2, num_particles))
    velocity = np.zeros(shape=(batch_size, 2, num_particles))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, :, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
          velocity[b, :, i * 10 + j] = (1 * (j - 4.5), -1 * (i - 4.5))
    input_state = sim.get_initial_state(position=position, velocity=velocity)

    memo = sim.run(1000, input_state)
    sim.visualize(memo, interval=5)

  def test_dilating_cube(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    num_particles = 100
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=1e-3,
        sess=sess,
        E=0.01)
    position = np.zeros(shape=(batch_size, 2, num_particles))
    velocity = np.zeros(shape=(batch_size, 2, num_particles))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, :, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
          velocity[b, :, i * 10 + j] = (0.5 * (i - 4.5), 0)
    input_state = sim.get_initial_state(position=position, velocity=velocity)

    memo = sim.run(100, input_state)
    sim.visualize(memo)

  def test_initial_gradients(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    N = 10
    num_particles = N * N
    steps = 10
    dt = 1e-3
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=dt,
        sess=sess)
    position = np.zeros(shape=(batch_size, 2, num_particles))
    youngs_modulus = np.zeros(shape=(batch_size, 1, num_particles))
    velocity_ph = tf.compat.v1.placeholder(shape=(2,), dtype=tf_precision)
    velocity = velocity_ph[None, :, None] + tf.zeros(
        shape=[batch_size, 2, num_particles], dtype=tf_precision)
    for b in range(batch_size):
      for i in range(N):
        for j in range(N):
          position[b, :, i * N + j] = ((i * 0.5 + 12.75) * dx,
                                    (j * 0.5 + 12.75) * dx)
    input_state = sim.get_initial_state(
        position=position, velocity=velocity, youngs_modulus=youngs_modulus)

    loss = tf.reduce_mean(input_tensor=sim.initial_state.center_of_mass()[:, 0])
    memo = sim.run(steps, input_state, initial_feed_dict={velocity_ph: [3, 2]})
    
    sim.set_initial_state(input_state)
    sym = sim.gradients_sym(loss=loss, variables=[velocity_ph])
    grad = sim.eval_gradients(sym, memo)
    self.assertAlmostEqualFloat32(grad[0][0], steps * dt)
    self.assertAlmostEqualFloat32(grad[0][1], 0)
    
  def test_gradients(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    N = 2
    num_particles = N * N
    steps = 30
    dt = 1e-2
    sim = Simulation(
      grid_res=(30, 30),
      dx=dx,
      num_particles=num_particles,
      gravity=gravity,
      dt=dt,
      batch_size=batch_size,
      E=1,
      sess=sess)
  
    position_ph = tf.compat.v1.placeholder(shape=(batch_size, 2, num_particles), dtype=tf.float32)
    velocity_ph = tf.compat.v1.placeholder(shape=(batch_size, 2, num_particles), dtype=tf.float32)
  
    position_val = np.zeros(shape=(batch_size, 2, num_particles))
    velocity_val = np.zeros(shape=(batch_size, 2, num_particles))
  
    F_val = np.zeros(shape=(batch_size, 2, 2, num_particles))
    F_val[:, 0, 0, :] = 0.5
    F_val[:, 0, 1, :] = 0.5
    F_val[:, 1, 0, :] = -0.5
    F_val[:, 1, 1, :] = 1
  
    for b in range(batch_size):
      for i in range(N):
        for j in range(N):
          position_val[b, :, i * N + j] = (0.5 + i * dx * 0.3, 0.5 + j * dx * 0.2)
  
    input_state = sim.get_initial_state(position=position_ph, velocity=velocity_ph, deformation_gradient=F_val)
  
    # loss = sim.initial_state.velocity[:, 1, 0]
    #loss = sim.initial_state.deformation_gradient[:, 1, 1, 0]
    loss = sim.initial_state.affine[:, 1, 0, 0]
  
    sim.set_initial_state(input_state)
    sym = sim.gradients_sym(loss=loss, variables=[position_ph, velocity_ph])
  
    def forward(pos, vel):
      memo = sim.run(steps, input_state, initial_feed_dict={velocity_ph: vel, position_ph: pos}, loss=loss)
      return memo.loss[0]
  
    memo = sim.run(steps, input_state, initial_feed_dict={velocity_ph: velocity_val, position_ph: position_val})
    sim.visualize(memo)
    grad = sim.eval_gradients(sym, memo)
    delta = 1e-3
    dim = 2
  
    for i in range(dim):
      for j in range(num_particles):
        position_val[0, i, j] += delta
        v1 = forward(position_val, velocity_val)
      
        position_val[0, i, j] -= 2 * delta
        v2 = forward(position_val, velocity_val)
      
        position_val[0, i, j] += delta
      
        g = (v1 - v2) / (2 * delta)
        print(g, grad[0][0, i, j])
        self.assertAlmostEqualFloat32(g, grad[0][0, i, j], clip=1e-2, relative_tol=4e-2)
  
    for i in range(dim):
      for j in range(num_particles):
        velocity_val[0, i, j] += delta
        v1 = forward(position_val, velocity_val)
      
        velocity_val[0, i, j] -= 2 * delta
        v2 = forward(position_val, velocity_val)
      
        velocity_val[0, i, j] += delta
      
        g = (v1 - v2) / (2 * delta)
        print(g, grad[1][0, i, j])
        self.assertAlmostEqualFloat32(g, grad[1][0, i, j], clip=1e-2, relative_tol=4e-2)


  def test_bc_gradients(self):
    batch_size = 1
    gravity = (0, -0)
    N = 10
    num_particles = N * N
    steps = 70
    dt = 1e-2
    res = (30, 30)

    goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')
    velocity_delta = np.zeros([batch_size, 2, num_particles])

    sim = Simulation(
      dt=dt,
      num_particles=num_particles,
      grid_res=res,
      gravity=gravity,
      m_p=1,
      V_p=1,
      E = 5,
      nu = 0.3,
      sess=sess)
    
    position = np.zeros(shape=(batch_size, num_particles, 2))

    velocity_ph = tf.compat.v1.placeholder(shape=(2,), dtype=tf.float32)
    velocity = velocity_ph[None, :, None] + tf.zeros(
      shape=[batch_size, 2, num_particles], dtype=tf.float32)

    part_size = N * N // 2
    velocity_part = tf.compat.v1.placeholder(shape = (2, ), dtype = tf.float32)
    velocity_p = velocity_part[None, :, None] + tf.zeros(shape = (batch_size, 2, part_size), dtype = tf.float32)
    velocity_2 = tf.concat([velocity_p,
                           tf.zeros(shape = (batch_size, 2, num_particles - part_size), dtype = tf.float32)],
                           axis = 2)
    velocity = velocity + velocity_2

    for b in range(batch_size):
      for i in range(N):
        for j in range(N):
          position[b, i * N + j] = ((i * 0.5 + 5) / 30,
                                    (j * 0.5 + 12.75) / 30)
          velocity_delta[b, :, i * N + j] = (float(j) / N - 0.5, 0.5 - float(i) / N)
          
    velocity = velocity + velocity_delta
    position = np.array(position).swapaxes(1, 2)

    sess.run(tf.compat.v1.global_variables_initializer())

    initial_state = sim.get_initial_state(
      position=position, velocity=velocity)

    final_position = sim.initial_state.center_of_mass()
    loss = tf.reduce_sum(input_tensor=(final_position - goal) ** 2)
    sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
    sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)

    sim.set_initial_state(initial_state = initial_state)

    sym = sim.gradients_sym(loss, variables = [velocity_part])

    goal_input = np.array([[0.7, 0.3]], dtype=np.float32)

    def forward(in_v, d_v):
      memo = sim.run(
        initial_state=initial_state,
        num_steps = steps,
        initial_feed_dict = {velocity_ph: in_v, velocity_part : d_v},
        iteration_feed_dict = {goal: goal_input},
        loss = loss)
      return memo.loss
      
    in_v = [0.2, -1]
    d_v = [0, 0]
    memo = sim.run(
      initial_state=initial_state,
      num_steps = steps,
      initial_feed_dict = {velocity_ph: in_v, velocity_part : d_v},
      iteration_feed_dict = {goal: goal_input},
      loss = loss)

    grad = sim.eval_gradients(sym, memo)
    sim.visualize(memo)
    print(grad)
    delta = 1e-4
    for i in range(2):
      d_v[i] += delta
      f1 = forward(in_v, d_v)
      d_v[i] -= 2 * delta
      f2 = forward(in_v, d_v)
      d_v[i] += delta
      print((f1 - f2) / (2 * delta))

if __name__ == '__main__':
  unittest.main()
