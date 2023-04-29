import sys
import math
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

evaluate = False
lr = 10
gamma = 0.0

sample_density = 15
group_num_particles = sample_density**3
goal_pos = np.array([1.4, 0.4, 0.5])
goal_range = np.array([0.0, 0.0, 0.0])
batch_size = 1

actuation_strength = 4


config = 'C'

exp = export.Export('crawler3d')

# Robot B
if config == 'B':
  num_groups = 7
  group_offsets = [(0, 0, 0), (0.5, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0), (2, 0, 0), (2.5, 0, 0)]
  group_sizes = [(0.5, 1, 1), (0.5, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (0.5, 1, 1), (0.5, 1, 1)]
  actuations = [0, 1, 5, 6]
  fixed_groups = []
  head = 3
  gravity = (0, -2, 0)



#TODO: N-Ped
#Robot C
else:

  num_leg_pairs = 2

  x = 3
  z = 3
  act_x = 0.5
  act_y = 0.35
  act_z = 1

  group_offsets = []
  for x_i in np.linspace(0, x - 2*act_x, num_leg_pairs):
    
    group_offsets += [(x_i, 0, 0)]
    group_offsets += [(x_i + act_x, 0, 0)]
    group_offsets += [(x_i, act_y, 0)]
    group_offsets += [(x_i + act_x, act_y, 0)]

    group_offsets += [(x_i + act_x, 0, act_z * 2)]
    group_offsets += [(x_i, 0, act_z * 2)]
    group_offsets += [(x_i + act_x, act_y, act_z * 2)]
    group_offsets += [(x_i, act_y, act_z * 2)]

  
  for i in range(3):
    group_offsets += [(i, 0, act_z)]
  num_groups = len(group_offsets)
      
  #group_offsets += [(0.0, 1.0, 0.0)]
  num_particles = group_num_particles * num_groups
  group_sizes = [(act_x, act_y, act_z)] * num_leg_pairs * 2 * 4 + [(1.0, 0.8, 1.0)] * int(x)
  actuations = list(range(8 * num_leg_pairs))
  fixed_groups = []
  head = 2 * 8 + 2
  gravity = (0, -2, 0)

#IPython.embed()


num_particles = group_num_particles * num_groups


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)

K = 8

# NN weights
W1 = tf.Variable(
    0.02 * tf.random.normal(shape=(len(actuations), K)),
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
    controller_inputs_backup = controller_inputs
    controller_inputs = []
    t = tf.ones(shape=(1, 1)) * 0.06 * tf.cast(state.step_count, dtype=tf.float32)
    for k in range(8):
      controller_inputs.append(tf.sin(t + 2 / K * k * math.pi))
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (batch_size, K), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (batch_size, K, 1)
    # Batch, 6 * num_groups, 1
    intermediate = tf.matmul(W1[None, :, :] +
                             tf.zeros(shape=[batch_size, 1, 1]), controller_inputs)
    # Batch, #actuations, 1
    assert intermediate.shape == (batch_size, len(actuations), 1)
    assert intermediate.shape[2] == 1
    intermediate = intermediate[:, :, 0]
    # Batch, #actuations
    actuation = tf.tanh(intermediate + b1[None, :]) * actuation_strength
    
    controller_inputs_backup = tf.concat(controller_inputs_backup, axis=1)
    controller_inputs_backup = controller_inputs_backup[:, :, None]
    
    debug = {'controller_inputs': controller_inputs_backup[:, :, 0], 'actuation': actuation}
    total_actuation = 0
    zeros = tf.zeros(shape=(batch_size, num_particles))
    for i, group in enumerate(actuations):
      act = actuation[:, i:i+1]
      assert len(act.shape) == 2
      mask = particle_mask_from_group(group)
      act = act * mask
      act = make_matrix3d(zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros, act)
      total_actuation = total_actuation + act
    return total_actuation, debug
  
  
  res = (60 + 100 * int(evaluate), 30, 30)
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
      E=25, damping=0.001 * evaluate)
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
                ) * scale + 0.2
            v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
                ) * scale + 0.1
            w = ((z + 0.5) / sample_density * group_sizes[i][2] + offset[2]
                 ) * scale + 0.1
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
  goal_train = [np.array(
    [[pos_x + (random.random() - 0.5) * gx,
      pos_y + (random.random() - 0.5) * gy,
      pos_z + (random.random() - 0.5) * gz
      ] for _ in range(batch_size)],
    dtype=np.float32) for __ in range(1)]

  vis_id = list(range(batch_size))
  random.shuffle(vis_id)

  # Optimization loop
  saver = tf.compat.v1.train.Saver()
  
  if evaluate:
    '''evaluate'''
    saver.restore(sess, "crawler3d_demo/0014/data.ckpt")
    tt = time.time()
    memo = sim.run(
      initial_state=initial_state,
      num_steps=1800,
      iteration_feed_dict={goal: goal_train[0]},
      loss=loss)
    print('forward', time.time() - tt)

    fn = 'crawler3d_demo/eval'
    sim.visualize(memo, batch=random.randrange(batch_size), export=None,
                  show=True, interval=5, folder=fn)
    return
    
  for e in range(100000):
    t = time.time()
    print('Epoch {:5d}, learning rate {}'.format(e, lr))

    loss_cal = 0.
    print('train...')
    for it, goal_input in enumerate(goal_train):
      tt = time.time()
      memo = sim.run(
          initial_state=initial_state,
          num_steps=400,
          iteration_feed_dict={goal: goal_input},
          loss=loss)
      print('forward', time.time() - tt)
      tt = time.time()
      grad = sim.eval_gradients(sym=sym, memo=memo)
      print('backward', time.time() - tt)

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
      fn = 'crawler3d_demo/{:04d}/'.format(e)
      saver.save(sess, "{}/data.ckpt".format(fn))
      sim.visualize(memo, batch=random.randrange(batch_size), export=None,
                    show=True, interval=5, folder=fn)

#exp.export()
    print('train loss {}'.format(loss_cal / len(goal_train)))
    
if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
