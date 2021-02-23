import sys
sys.path.append('..')

import random
import os
import numpy as np
from simulation import Simulation, get_bounding_box_bc
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.layers as ly
from vector_math import *
import export 
import IPython

lr = 0.03
gamma = 0.0

sample_density = 20
group_num_particles = sample_density**2
goal_pos = np.array([1.4, 0.4])
goal_range = np.array([0.0, 0.00])
batch_size = 1

config = 'B'

exp = export.Export('walker_video')

# Robot B
num_groups = 1
group_offsets = [(1, 1)]
group_sizes = [(0.5, 1)]
actuations = [0]
fixed_groups = []
head = 0
gravity = (0, 0)

num_particles = group_num_particles * num_groups


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)


# NN weights
W1 = tf.Variable(
    0.02 * tf.random.normal(shape=(len(actuations), 6 * len(group_sizes))),
    trainable=True)
b1 = tf.Variable([0.0] * len(actuations), trainable=True)


def main(sess):
  t = time.time()

  goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  # Define your controller here
  def controller(state):
    actuation = 2 * np.ones(shape=(batch_size, num_groups)) * tf.sin(0.1 * tf.cast(state.get_evaluated()['step_count'], tf.float32))
    total_actuation = 0
    zeros = tf.zeros(shape=(batch_size, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1] * 2
      assert len(act.shape) == 2
      mask = particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      act = make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + act
    return total_actuation, 1
  
  res = (80, 40)
  bc = get_bounding_box_bc(res)
  
  sim = Simulation(
      dt=0.005,
      num_particles=num_particles,
      grid_res=res,
      dx=1.0 / res[1],
      gravity=gravity,
      controller=controller,
      batch_size=batch_size,
      bc=bc,
      sess=sess,
      scale=20)
  print("Building time: {:.4f}s".format(time.time() - t))

  s = head * 6
  
  initial_positions = [[] for _ in range(batch_size)]
  initial_velocity = np.zeros(shape=(batch_size, 2, num_particles))
  for b in range(batch_size):
    c = 0
    for i, offset in enumerate(group_offsets):
      c += 1
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
              ) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
              ) * scale + 0.1
          initial_positions[b].append([u, v])
          initial_velocity[0, :, c] = (2 * (y - sample_density / 2), -2 * (x - sample_density / 2))
  assert len(initial_positions[0]) == num_particles
  initial_positions = np.array(initial_positions).swapaxes(1, 2)

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=np.array(initial_positions), youngs_modulus=10, velocity=initial_velocity)

  sim.set_initial_state(initial_state=initial_state)
  
  gx, gy = goal_range
  pos_x, pos_y = goal_pos
  goal_train = [np.array(
    [[pos_x + (random.random() - 0.5) * gx,
      pos_y + (random.random() - 0.5) * gy] for _ in range(batch_size)],
    dtype=np.float32) for __ in range(1)]

  vis_id = list(range(batch_size))
  random.shuffle(vis_id)

  # Optimization loop
  for i in range(100000):
    t = time.time()
    print('Epoch {:5d}, learning rate {}'.format(i, lr))

    print('train...')
    for it, goal_input in enumerate(goal_train):
      tt = time.time()
      memo = sim.run(
          initial_state=initial_state,
          num_steps=400,
          iteration_feed_dict={goal: goal_input},
          )
      print('forward', time.time() - tt)
      tt = time.time()
      print('backward', time.time() - tt)

      sim.visualize(memo, batch=random.randrange(batch_size), export=exp,
                    show=True, interval=4)
    #exp.export()

if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
