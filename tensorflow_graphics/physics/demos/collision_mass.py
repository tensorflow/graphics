import sys
sys.path.append('..')

import random
import time
from simulation import Simulation, get_bounding_box_bc
from time_integration import UpdatedSimulationState
import tensorflow as tf
import numpy as np
from IPython import embed

batch_size = 1
gravity = (0, 0)
N = 40
group_particles = N * N
num_particles = group_particles * 2
steps = 300
dt = 3e-3
goal_range = 0.15
res = (100, 100)
bc = get_bounding_box_bc(res)

lr = 1

def main(sess):
  goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')
  sim = Simulation(
      dt=dt,
      num_particles=num_particles,
      grid_res=res,
      bc=bc,
      gravity=gravity,
      sess=sess)
  position = np.zeros(shape=(batch_size, num_particles, 2))

  velocity_ph = tf.compat.v1.placeholder(shape = (2, ), dtype = tf.float32)
  velocity_1 = velocity_ph[None, :, None] + tf.zeros(
      shape=[batch_size, 2, group_particles], dtype=tf.float32)
  velocity_2 = tf.zeros(shape=[batch_size, 2, group_particles], dtype=tf.float32)
  velocity = tf.concat([velocity_1, velocity_2], axis = 2)

  mass_ph = tf.Variable([1.], trainable = True)
  mass_1 = mass_ph[None, None, :] + tf.zeros(
      shape = [batch_size, 1, group_particles], dtype = tf.float32)
  mass_2 = tf.ones(shape = [batch_size, 1, group_particles], dtype = tf.float32)
  mass = tf.concat([mass_1, mass_2], axis = 2)

  for b in range(batch_size):
    for i in range(group_particles):
      x, y = 0, 0
      while (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.25:
        x, y = random.random(), random.random()
      position[b, i] = ((x * 2 + 3) / 30,
                        (y * 2 + 12.25) / 30)

    for i in range(group_particles):
      x, y = 0, 0
      while (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.25:
        x, y = random.random(), random.random()
      position[b, i + group_particles] = ((x * 2 + 10) / 30,
                                          (y * 2 + 12.75) / 30)
  position = np.array(position).swapaxes(1, 2)

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=position, velocity=velocity, particle_mass = mass)

  final_position = sim.initial_state.center_of_mass(group_particles, None)
  loss = tf.reduce_sum(input_tensor=(final_position - goal) ** 2) ** 0.5
  # loss = tf.reduce_sum(tf.abs(final_position - goal))
  sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
  sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)

  trainables = [mass_ph]
  sim.set_initial_state(initial_state = initial_state)

  sym = sim.gradients_sym(loss, variables = trainables)

  goal_input = np.array(
          [[0.68, 0.52]],
    dtype=np.float32)

  for i in range(1000000):
    t = time.time()
    memo = sim.run(
        initial_state = initial_state,
        num_steps = steps,
        initial_feed_dict = {velocity_ph: [0.5, 0]},
        iteration_feed_dict = {goal: goal_input},
        loss = loss)
    grad = sim.eval_gradients(sym, memo)
    sim.visualize(memo, interval=5, folder='sysid_demo/iteration{:04d}'.format(i))
    print('mass', mass_ph.eval())
    gradient_descent = [
        v.assign(v - lr * g) for v, g in zip(trainables, grad)
    ]
    sess.run(gradient_descent)
    print('iter {:5d} time {:.3f} loss {:.4f}'.format(
        i, time.time() - t, memo.loss))
    
if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
