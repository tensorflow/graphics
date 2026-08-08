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
gravity = (0, -1)
N = 10
group_size = np.array([[10, 20], [10, 20], [90, 10], [10, 10]])
group_position = np.array([[5, 2.5], [20, 2.5], [2.5, 8.5], [12.5, 12]])
num_particles = (group_size[:, 0] * group_size[:, 1]).sum()
steps = 200
dt = 5e-3
goal_range = 0.15
res = (30, 30)
bc = get_bounding_box_bc(res)

lr = 10

def main(sess):
  
  goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  sim = Simulation(
      dt=dt,
      num_particles=num_particles,
      grid_res=res,
      bc=bc,
      gravity=gravity,
      sess=sess)

  '''
  youngs_modulus = np.ones(shape=(batch_size, num_particles, 1))
  youngs_modulus[:, : 400] = 100
  youngs_modulus[:, 400: -100] = 10
  youngs_modulus[:, -100: ] = 10
  '''
  ym_var = tf.Variable([5.], trainable = True)
  ym_1 = tf.constant(100., shape = [1, 1, 400], dtype = tf.float32)
  ym_2 = ym_var + tf.zeros(shape = [1, 1, 900], dtype = tf.float32)
  ym_3 = tf.constant(10., shape = [1, 1, 100], dtype = tf.float32)
  youngs_modulus = tf.concat([ym_1, ym_2, ym_3], axis = 2)


  velocity = np.zeros(shape = (batch_size, 2, num_particles))
  position = np.zeros(shape = (batch_size, num_particles, 2)) #swap axis later

  for b in range(batch_size):
    cnt = 0
    for group, pos in zip(group_size, group_position):
      n, m = group
      x, y = pos
      for i in range(n):
        for j in range(m):
          position[b][cnt] = ((i * 0.25 + x) / res[0],
                              (j * 0.25 + y) / res[1])
          cnt += 1
  position = np.swapaxes(position, 1, 2)

  mass = np.ones(shape = (batch_size, 1, num_particles))
  mass[:, :, :400] = 20

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=position, velocity=velocity, 
      particle_mass=mass, youngs_modulus=youngs_modulus)

  final_position = sim.initial_state.center_of_mass(-100, None)
  loss = tf.reduce_sum(input_tensor=tf.abs(final_position - goal)[:, 1])
  # loss = tf.reduce_sum(tf.abs(final_position - goal))
  sim.add_point_visualization(pos = final_position, color = (1, 0, 0), radius = 3)
  sim.add_point_visualization(pos = goal, color = (0, 1, 0), radius = 3)

  trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state = initial_state)

  sym = sim.gradients_sym(loss, variables = trainables)

  goal_input = np.array(
          [[0.5, 0.4]],
    dtype=np.float32)

  log = tf.compat.v1.Print(ym_var, [ym_var], 'youngs_modulus:')

  for i in range(1000000):
    t = time.time()
    memo = sim.run(
        initial_state = initial_state, 
        num_steps = steps,
        iteration_feed_dict = {goal: goal_input},
        loss = loss)
    grad = sim.eval_gradients(sym, memo)
    #if i % 5 == 0: # True: # memo.loss < 0.01:
    sim.visualize(memo, interval = 2, show=True)
    gradient_descent = [
        v.assign(v - lr * g) for v, g in zip(trainables, grad)
    ]
    sess.run(gradient_descent)
    print('iter {:5d} time {:.3f} loss {:.4f}'.format(
        i, time.time() - t, memo.loss))
    log.eval()
    
if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
