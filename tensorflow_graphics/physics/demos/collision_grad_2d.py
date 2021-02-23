import sys
sys.path.append('..')

import random
import time
from simulation import Simulation, get_bounding_box_bc
from simulation import tf_precision, np_precision
import tensorflow as tf
import numpy as np
from IPython import embed

batch_size = 1
gravity = (0, 0)
N = 10
group_particles = N * N * 2
num_particles = group_particles * 2
steps = 100
dt = 5e-3
goal_range = 0.15
res = (100, 100)
bc = get_bounding_box_bc(res)

lr = 5e-1

def main(sess):

  random.seed(100)

  goal = tf.compat.v1.placeholder(dtype=tf_precision, shape=[batch_size, 2], name='goal')

  sim = Simulation(
    dt=dt,
    num_particles=num_particles,
    grid_res=res,
    bc=bc,
    gravity=gravity,
    E=1,
    m_p=1,
    V_p=1,
    sess=sess,
    part_size = 20)
  position = np.zeros(shape=(batch_size, 2, num_particles), dtype = np_precision)
  velocity_delta = np.zeros(shape=(batch_size, 2, num_particles), dtype = np_precision)

  # velocity_ph = tf.Variable([0.4, 0.05], trainable = True)
  velocity_ph = tf.compat.v1.placeholder(dtype = tf_precision, shape = [batch_size, 2], name = 'velocity')
  velocity_1 = velocity_ph[:, :, None] + tf.zeros(
    shape=[batch_size, 2, group_particles], dtype=tf_precision)
  velocity_2 = tf.zeros(shape=[batch_size, 2, group_particles], dtype=tf_precision)
  velocity = tf.concat([velocity_1, velocity_2], axis = 2)

  for b in range(batch_size):
    for i in range(group_particles):
      x, y = 0, 0
      while (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.25:
        x, y = random.random(), random.random()
      position[b, :, i] = ((x * 2 + 3) / 30,
                           (y * 2 + 12.75) / 30)
      velocity_delta[b, :, i] = (y - 0.5, -x + 0.5)

    for i in range(group_particles):
      x, y = 0, 0
      while (x - 0.5) ** 2 + (y - 0.5) ** 2 > 0.25:
        x, y = random.random(), random.random()
      position[b, :, i + group_particles] = ((x * 2 + 10) / 30,
                                             (y * 2 + 12.75) / 30)


  velocity += velocity_delta

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(position=position, velocity=velocity)

  final_position = sim.initial_state.center_of_mass(group_particles, None)
  loss = tf.reduce_sum(input_tensor=(final_position - goal) ** 2)
  # loss = tf.reduce_sum(tf.abs(final_position - goal))
  sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
  sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)

  trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state = initial_state)

  sym = sim.gradients_sym(loss, variables = [velocity_ph])

  goal_input = np.array([[0.7, 0.3]], dtype=np_precision)
  v = np.array([[0.4, 0.05]], dtype = np_precision)

  delta = 1e-3
  for i in range(1000000):
    memo = sim.run(
      initial_state = initial_state,
      num_steps = steps,
      initial_feed_dict = {velocity_ph: v},
      iteration_feed_dict = {goal: goal_input},
      loss = loss)
    print(memo.loss)
    sim.visualize(memo, interval=10)
    grad = sim.eval_gradients(sym, memo)
    print('grad', grad)
    for d in range(2):
      v[:, d] += delta
      memo = sim.run(
        initial_state = initial_state,
        num_steps = steps,
        initial_feed_dict = {velocity_ph: v},
        iteration_feed_dict = {goal: goal_input},
        loss = loss)
      loss1 = memo.loss
      v[:, d] -= 2 * delta
      memo = sim.run(
        initial_state = initial_state,
        num_steps = steps,
        initial_feed_dict = {velocity_ph: v},
        iteration_feed_dict = {goal: goal_input},
        loss = loss)
      loss2 = memo.loss
      v[:, d] += delta
      print((loss1 - loss2) / 2.0 / delta)
    break

if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
