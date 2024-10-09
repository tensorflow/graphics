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

lr = 1
gamma = 0.0

sample_density = 15
group_num_particles = sample_density**3
goal_pos = np.array([1, 1.17, 1])
goal_range = np.array([0.6, 0.0, 0.6])
batch_size = 1

actuation_strength = 40


config = 'C'

act_x = 0.5
act_y = 1
act_z = 0.5

x = 1
z = 1


group_offsets = []
num_links = 3

for i in range(num_links):
  y = act_y * 2 * i
  group_offsets += [(0, y, 0)]
  group_offsets += [(0 + act_x, y, 0)]
  group_offsets += [(0, y, act_z)]
  group_offsets += [(0 + act_x, y, act_z)]

for i in range(num_links):
  group_offsets += [(0, act_y * (2 * i + 1), 0)]
  
num_groups = len(group_offsets)

num_particles = group_num_particles * num_groups

group_sizes = [(act_x, act_y, act_z)] * 4 * num_links

for i in range(num_links):
  group_sizes += [(1, 1, 1)]
  
actuations = list(range(4 * num_links))
gravity = (0, 0, 0)
head = len(group_offsets) - 1

num_particles = group_num_particles * num_groups


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)


# NN weights
W1 = tf.Variable(
    0.02 * tf.random.normal(shape=(len(actuations), 9 * len(group_sizes))),
    trainable=True)
b1 = tf.Variable([0.0] * len(actuations), trainable=True)


def main(sess):
  t = time.time()

  goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 3], name='goal')

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
      controller_inputs.append((goal - goal_pos) / np.maximum(goal_range, 1e-5))
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (batch_size, 9 * num_groups), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (batch_size, 9 * num_groups, 1)
    # Batch, 6 * num_groups, 1
    intermediate = tf.matmul(W1[None, :, :] +
                             tf.zeros(shape=[batch_size, 1, 1]), controller_inputs)
    # Batch, #actuations, 1
    assert intermediate.shape == (batch_size, len(actuations), 1)
    assert intermediate.shape[2] == 1
    intermediate = intermediate[:, :, 0]
    # Batch, #actuations
    actuation = tf.tanh(intermediate + b1[None, :]) * actuation_strength
    debug = {'controller_inputs': controller_inputs[:, :, 0], 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(batch_size, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1]
      assert len(act.shape) == 2
      mask = particle_mask_from_group(group)
      act = act * mask
      act = make_matrix3d(zeros, zeros, zeros, zeros, act, zeros, zeros, zeros, zeros)
      total_actuation = total_actuation + act
    return total_actuation, debug
  
  res = (60, 90, 60)
  bc = get_bounding_box_bc(res)
  # stick it to the ground
  bc[0][:, :, :5, :] = -1
  
  sim = Simulation(
      dt=0.0015,
      num_particles=num_particles,
      grid_res=res,
      dx=1.0 / 30,
      gravity=gravity,
      controller=controller,
      batch_size=batch_size,
      bc=bc,
      sess=sess,
      E=150)
  print("Building time: {:.4f}s".format(time.time() - t))

  final_state = sim.initial_state['debug']['controller_inputs']
  s = head * 9
  
  final_position = final_state[:, s:s+3]
  final_velocity = final_state[:, s + 3: s + 6]
  loss1 = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=(final_position - goal) ** 2, axis = 1))
  loss2 = tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=final_velocity ** 2, axis = 1)) 

  loss = loss1 + gamma * loss2

  initial_positions = [[] for _ in range(batch_size)]
  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          for z in range(sample_density):
            scale = 0.2
            u = ((x + 0.5) / sample_density * group_sizes[i][0] + offset[0]
                ) * scale + 0.9
            v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
                ) * scale + 0.1
            w = ((z + 0.5) / sample_density * group_sizes[i][2] + offset[2]
                 ) * scale + 0.9
            initial_positions[b].append([u, v, w])
  assert len(initial_positions[0]) == num_particles
  initial_positions = np.array(initial_positions).swapaxes(1, 2)

  sess.run(tf.compat.v1.global_variables_initializer())

  initial_state = sim.get_initial_state(
      position=np.array(initial_positions), youngs_modulus=10)

  trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
  sim.set_initial_state(initial_state=initial_state)
  
  sym = sim.gradients_sym(loss, variables=trainables)

  gx, gy, gz = goal_range
  pos_x, pos_y, pos_z = goal_pos

  vis_id = list(range(batch_size))
  random.shuffle(vis_id)

  # Optimization loop
  for e in range(100000):
    t = time.time()
    goal_train = [np.array(
      [[pos_x + (random.random() - 0.5) * gx,
        pos_y + (random.random() - 0.5) * gy,
        pos_z + (random.random() - 0.5) * gz
        ] for _ in range(batch_size)],
      dtype=np.float32) for __ in range(10)]
    print('goal', goal_train)
    print('Epoch {:5d}, learning rate {}'.format(e, lr))

    loss_cal = 0.
    print('train...')
    for it, goal_input in enumerate(goal_train):
      tt = time.time()
      memo = sim.run(
          initial_state=initial_state,
          num_steps=200,
          iteration_feed_dict={goal: goal_input},
          loss=loss)
      tt = time.time()
      grad = sim.eval_gradients(sym=sym, memo=memo)

      for i, g in enumerate(grad):
        print(i, np.mean(np.abs(g)))
      grad = [np.clip(g, -1, 1) for g in grad]


      gradient_descent = [
          v.assign(v - lr * g) for v, g in zip(trainables, grad)
      ]
      sess.run(gradient_descent)
      print('Iter {:5d} time {:.3f} loss {}'.format(
          it, time.time() - t, memo.loss))
      loss_cal = loss_cal + memo.loss
      folder = 'arm3d_demo/{:04d}/'.format(e * len(goal_train) + it)
      sim.visualize(memo, batch=random.randrange(batch_size), export=None,
                    show=True, interval=3, folder=folder)
      with open(os.path.join(folder, 'target.txt'), 'w') as f:
        goal_input = goal_input[0]
        print(goal_input[0], goal_input[1], goal_input[2], file=f)
    #exp.export()
    print('train loss {}'.format(loss_cal / len(goal_train)))
    
if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
