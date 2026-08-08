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

sample_density = 40
group_num_particles = sample_density**2
goal_pos = np.array([0.5, 0.6])
goal_range = np.array([0.1, 0.1])
batch_size = 1
actuation_strength = 4
multi_target = True

config = 'B'

exp = export.Export('finger_data')

# Robot B
num_groups = 3
group_offsets = [(1, 0), (1.5, 0), (1, 2)]
group_sizes = [(0.5, 2), (0.5, 2), (1, 1)]
actuations = [0, 1]
head = 2
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


sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4


sess = tf.compat.v1.Session(config=sess_config)

goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')
actuation = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, len(actuations)], name='act')

def generate_sim():
  #utility function for ppo
  t = time.time()

  

  # Define your controller here
  
  def controller(state):
    controller_inputs = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles,
                           (i + 1) * group_num_particles)[:, None, :] * (
                               1.0 / group_num_particles)
      pos = tf.reduce_sum(input_tensor=mask * state.position, axis=2, keepdims=False)
      vel = tf.reduce_sum(input_tensor=mask * state.velocity, axis=2, keepdims=False)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      in_goal = goal
      if multi_target:
        in_goal = (goal - goal_pos) / goal_range
      controller_inputs.append((goal - 0.5))
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (batch_size, 6 * num_groups), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (batch_size, 6 * num_groups, 1)
    # Batch, 6 * num_groups, 1
      #IPython.embed()
    
    debug = {'controller_inputs': controller_inputs[:, :, 0], 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(batch_size, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1]
      assert len(act.shape) == 2
      mask = particle_mask_from_group(group)
      act = act * mask
      # First PK stress here
      act = make_matrix2d(zeros, zeros, zeros, act)
      # Convert to Kirchhoff stress
      total_actuation = total_actuation + act
    return total_actuation, debug
  
  
  res = (40, 40)
  bc = get_bounding_box_bc(res)
  bc[0][:, :, :7] = -1 # Sticky
  bc[1][:, :, :7] = 0 # Sticky
  dt = 0.005  

  sim = Simulation(
      dt=dt,
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

  final_state = sim.initial_state['debug']['controller_inputs']
  s = head * 6
  
  final_position = final_state[:, s:s+2]
  final_velocity = final_state[:, s + 2: s + 4]
  loss1 = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=(final_position - goal) ** 2, axis = 1))
  loss2 = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=final_velocity ** 2, axis = 1)) 
  
  
  loss_x = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=final_position[0, 0]))
  loss_y = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=final_position[0, 1]))
  
  
  loss_obs = final_state
  loss_fwd = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=final_state[:, s+2:s + 3], axis=1)) * dt

  #loss = loss_fwd 
  vel = tf.squeeze(ly.flatten(final_velocity))
  dire = tf.squeeze(ly.flatten(goal - final_position))
  loss = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.tensordot(vel, dire / tf.norm(tensor=dire), 1))) * dt
  

  initial_positions = [[] for _ in range(batch_size)]
  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
              ) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
              ) * scale + 0.1
          initial_positions[b].append([u, v])
  assert len(initial_positions[0]) == num_particles
  initial_positions = np.array(initial_positions).swapaxes(1, 2)

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=np.array(initial_positions), youngs_modulus=10)

  #trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  
  sim.add_point_visualization(pos=goal, color=(0, 1, 0), radius=3)
  sim.add_vector_visualization(pos=final_position, vector=final_velocity, color=(0, 0, 1), scale=50)
  
  
  return initial_state, sim, loss, loss_obs
    
#if __name__ == '__main__':
  
