import sys
sys.path.append('..')

import random
import time
from simulation import Simulation, get_bounding_box_bc
from time_integration import UpdatedSimulationState
import tensorflow as tf
import numpy as np
from IPython import embed
import export

exp = export.Export('rolling_acc')

def main(sess):
  batch_size = 1
  gravity = (0, -1)
  # gravity = (0, 0)
  N = 5
  dR = 0.2
  R = (N - 1) * dR
  dC = 1.6
  num_particles = int(((N - 1) * dC + 1) ** 2)
  steps = 1000
  dt = 5e-3
  goal_range = 0.15
  res = (45, 30)
  bc = get_bounding_box_bc(res)

  lr = 1e-2
  
  goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  def F_controller(state):
    F = state.position - state.center_of_mass()[:, :, None]
    F = tf.stack([F[:, 1], -F[:, 0]], axis = 1)
    # T = tf.cast(state.step_count // 100 % 2, dtype = tf.float32) * 2 - 1
    return F * 10#  * T

  sim = Simulation(
      dt=dt,
      num_particles=num_particles,
      grid_res=res,
      bc=bc,
      gravity=gravity,
      m_p=1,
      V_p=1,
      E = 10,
      nu = 0.3,
      sess=sess,
      use_visualize = True,
      F_controller = F_controller)
  position = np.zeros(shape=(batch_size, num_particles, 2))

  # velocity_ph = tf.constant([0.2, 0.3])
  velocity_ph = tf.constant([0, 0], dtype = tf.float32)
  velocity = velocity_ph[None, :, None] + tf.zeros(
      shape=[batch_size, 2, num_particles], dtype=tf.float32)
  random.seed(123)
  for b in range(batch_size):
    dx, dy = 5, 4
    cnt = 0
    las = 0
    for i in range(N):
      l = int((dC * i + 1) ** 2)
      l, las = l - las, l
      print(l)
      dth = 2 * np.pi / l
      dr = R / (N - 1) * i
      theta = np.pi * 2 * np.random.random()
      for j in range(l):
        theta += dth
        x, y = np.cos(theta) * dr, np.sin(theta) * dr
        position[b, cnt] = ((dx + x) / 30, (dy + y) / 30)
        cnt += 1

  position = np.array(position).swapaxes(1, 2)

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=position, velocity=velocity)

  final_position = sim.initial_state.center_of_mass()
  loss = tf.reduce_sum(input_tensor=(final_position - goal) ** 2)
  sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
  sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)

  trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state = initial_state)

  sym = sim.gradients_sym(loss, variables = trainables)

  goal_input = np.array([[0.7, 0.3]], dtype=np.float32)
  

  memo = sim.run(
      initial_state = initial_state, 
      num_steps = steps,
      iteration_feed_dict = {goal: goal_input},
      loss = loss)

  if True:
    sim.visualize(memo, show = True, interval = 2)
  else:
    sim.visualize(memo, show = False, interval = 1, export = exp)
    
if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
