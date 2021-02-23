import random
import os
from simulation import Simulation, get_bounding_box_bc
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.contrib.layers as ly
from vector_math import *
import IPython
import copy

import pygmo as pg
import pygmo_plugins_nonfree as ppnf

np.random.seed(326)

def flatten_vectors(vectors):
  return tf.concat([tf.squeeze(ly.flatten(vector)) for vector in vectors], 0)

lr = 1.0


goal_range = 0.0
batch_size = 1
actuation_strength = 8


use_pygmo = True


num_steps = 800

iter_ = 0

# Finger
num_links = 2
num_acts = int(num_steps // num_links) #TBH this is just to keep the number of variables tame
sample_density = int(20 // (np.sqrt(num_links)))
group_num_particles = sample_density**2
group_sizes = []
group_offsets = []
actuations = []
group_size = [(0.5, 2.0 / num_links), (0.5, 2.0 / num_links), (1, 1.0 / num_links)]
for i in range(num_links):
  group_offsets += [(1, (group_size[0][1] + group_size[2][1])*i ), (1.5, (group_size[1][1] + group_size[2][1])*i), (1, (group_size[0][1] + group_size[2][1])*i + group_size[0][1] )]
  group_sizes += copy.deepcopy(group_size)
  actuations += [0  + 3*i, 1 + 3*i]
num_groups = len(group_sizes)


head = num_groups - 1
gravity = (0, 0)


num_particles = group_num_particles * num_groups
num_actuators = len(actuations)


def particle_mask(start, end):
  r = tf.range(0, num_particles)
  return tf.cast(tf.logical_and(start <= r, r < end), tf.float32)[None, :]


def particle_mask_from_group(g):
  return particle_mask(g * group_num_particles, (g + 1) * group_num_particles)



actuation_seq = tf.Variable(1.0 * tf.random.normal(shape=(1, num_acts, num_actuators), dtype=np.float32), trainable=True)

def step_callback(dec_vec):
  pass




def main(sess):
  t = time.time()

  goal = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 2], name='goal')

  # Define your controller here
  def controller(state):    
    controller_inputs = []
    for i in range(num_groups):
      mask = particle_mask(i * group_num_particles,
                           (i + 1) * group_num_particles)[:, None, :] * (
                               1.0 / group_num_particles)
      pos = tf.reduce_sum(input_tensor=mask * state.position, axis=2, keepdims=False)
      vel = tf.reduce_sum(input_tensor=mask * state.velocity, axis=2, keepdims=False)
      accel = tf.reduce_sum(input_tensor=mask * state.acceleration, axis=2, keepdims=False)
      controller_inputs.append(pos)
      controller_inputs.append(vel)
      controller_inputs.append(goal)
      controller_inputs.append(accel)
    # Batch, dim
    controller_inputs = tf.concat(controller_inputs, axis=1)
    assert controller_inputs.shape == (batch_size, 8 * num_groups), controller_inputs.shape
    controller_inputs = controller_inputs[:, :, None]
    assert controller_inputs.shape == (batch_size, 8 * num_groups, 1)

    actuation = tf.expand_dims(actuation_seq[0, (state.step_count - 1) // (num_steps // num_acts), :], 0)
    debug = {'controller_inputs': controller_inputs[:, :, 0], 'actuation': actuation, 'acceleration': state.acceleration, 'velocity' : state.velocity}
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
  
  res = (30, 30)
  bc = get_bounding_box_bc(res)
  

  bc[0][:, :, :5] = -1 # Sticky
  bc[1][:, :, :5] = 0 # Sticky

  sim = Simulation(
      dt=0.0025,
      num_particles=num_particles,
      grid_res=res,
      gravity=gravity,
      controller=controller,
      batch_size=batch_size,
      bc=bc,
      sess=sess)
  print("Building time: {:.4f}s".format(time.time() - t))

  final_state = sim.initial_state['debug']['controller_inputs']
  final_acceleration = sim.initial_state['debug']['acceleration']
  final_velocity_all = sim.initial_state['debug']['velocity']
  s = head * 8
  
  final_position = final_state[:, s:s+2]
  final_velocity = final_state[:, s + 2: s + 4]
  final_accel = final_state[:, s + 6: s + 8]
  gamma = 0.0
  loss_position = tf.reduce_sum(input_tensor=(final_position - goal) ** 2)
  loss_velocity = tf.reduce_mean(input_tensor=final_velocity_all ** 2) / 10.0
  loss_act = tf.reduce_sum(input_tensor=actuation_seq ** 2.0) / 10000.0
  loss_zero = tf.Variable(0.0, trainable=False)
  
  #loss_accel = tf.reduce_mean(final_acceleration ** 2.0) / 10000.0
  loss_accel = loss_zero
  #IPython.embed()
  
  
  
  #acceleration_constraint = tf.reduce_sum(final_acceleration, axis=1)

  
  initial_positions = [[] for _ in range(batch_size)]
  for b in range(batch_size):
    for i, offset in enumerate(group_offsets):
      for x in range(sample_density):
        for y in range(sample_density):
          scale = 0.2
          u = ((x +0.5) / sample_density * group_sizes[i][0] + offset[0]
              ) * scale + 0.2
          v = ((y + 0.5) / sample_density * group_sizes[i][1] + offset[1]
              ) * scale + 0.1
          initial_positions[b].append([u, v])
  assert len(initial_positions[0]) == num_particles
  initial_positions = np.array(initial_positions).swapaxes(1, 2)

  youngs_modulus =tf.Variable(10.0 * tf.ones(shape = [1, 1, num_particles], dtype = tf.float32), trainable=True)
  initial_state = sim.get_initial_state(
      position=np.array(initial_positions), youngs_modulus=tf.identity(youngs_modulus))
      
  trainables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
 
  
  
  sess.run(tf.compat.v1.global_variables_initializer())
  
  sim.set_initial_state(initial_state=initial_state)
  
  sym_pos = sim.gradients_sym(loss_position, variables=trainables)
  sym_vel = sim.gradients_sym(loss_velocity, variables=trainables)
  sym_act = sim.gradients_sym(loss_act, variables=trainables)
  sym_zero = sim.gradients_sym(loss_zero, variables=trainables)
  sym_accel = sim.gradients_sym(loss_accel, variables=trainables)
  
  
  #sym_acc = [sim.gradients_sym(acceleration, variables=trainables) for acceleration in acceleration_constraint]
  #sym_acc = tf.map_fn(lambda x : sim.gradients_sym(x, variables=trainables), acceleration_constraint)
  #acc_flat = flatten_vectors([final_acceleration])
  #sym_acc = tf.map_fn((lambda x : sim.gradients_sym(x, variables=trainables)), acc_flat)
  #IPython.embed()
  
  sim.add_point_visualization(pos=goal, color=(0, 1, 0), radius=3)
  sim.add_vector_visualization(pos=final_position, vector=final_velocity, color=(0, 0, 1), scale=50)
 
  sim.add_point_visualization(pos=final_position, color=(1, 0, 0), radius=3)
  


  goal_input = np.array(
  [[0.7 + (random.random() - 0.5) * goal_range * 2,
    0.5 + (random.random() - 0.5) * goal_range] for _ in range(batch_size)],
  dtype=np.float32)
 
  

  def eval_sim(loss_tensor, sym_, need_grad=True):
    memo = sim.run(
        initial_state=initial_state,
        num_steps=num_steps,
        iteration_feed_dict={goal: goal_input},
        loss=loss_tensor)
    if need_grad:
      grad = sim.eval_gradients(sym=sym_, memo=memo)
    else:
      grad = None
    return memo.loss, grad, memo
  
  def flatten_trainables():
    return tf.concat([tf.squeeze(ly.flatten(trainable)) for trainable in trainables], 0)
    
  
    
  def assignment_run(xs):
    sess.run([trainable.assign(x) for x, trainable in zip(xs, trainables)])
  
  
        
  
  
  t = time.time()
    
  #loss_val, grad, memo = eval_sim(loss_position, sym_pos)
  
  #IPython.embed()
  
  
  #Begin optimization

  
  def assignment_helper(x):
    assignments = []
    idx = 0
    x = x.astype(np.float32)
    for v in trainables:
      #first, get count:
      var_cnt = tf.size(input=v).eval()
      assignments += [v.assign(tf.reshape(x[idx:idx+var_cnt],v.shape))]
      idx += var_cnt
    sess.run(assignments)
    
  class RobotProblem:
    def __init__(self, use_act):
      self.use_act = use_act
  
    goal_ball = 0.0001
    def fitness(self, x):      
      assignment_helper(x)
      if self.use_act:
        loss_act_val, _, _ = eval_sim(loss_act, sym_act, need_grad=False)
      else:
        loss_act_val, _, _ = eval_sim(loss_zero, sym_zero, need_grad=False)
      loss_pos_val, _, _ = eval_sim(loss_position, sym_pos, need_grad=False)
      loss_accel_val, _, _ = eval_sim(loss_accel, sym_accel, need_grad=False)
      c1, _, memo = eval_sim(loss_velocity, sym_vel, need_grad=False)        
      global iter_
      sim.visualize(memo, show = False, folder = "arm_log/it{:04d}".format(iter_))
      iter_ += 1
      print('loss pos', loss_pos_val)
      print('loss vel', c1)
      print('loss accel', loss_accel_val)
      #IPython.embed()
      return [loss_act_val.astype(np.float64), loss_pos_val.astype(np.float64) - self.goal_ball, c1.astype(np.float64) - self.goal_ball, loss_accel_val.astype(np.float64) - self.goal_ball]
      

    def get_nic(self):
      return 3
    def get_nec(self):
      return 0
      
    def gradient(self, x):
      assignment_helper(x)
      _, grad_position, _ = eval_sim(loss_position, sym_pos)
      _, grad_velocity, _ = eval_sim(loss_velocity, sym_vel)
      _, grad_accel, _ = eval_sim(loss_accel, sym_accel)
      if self.use_act:
        _, grad_act, _ = eval_sim(loss_act, sym_act)
      else:
        _, grad_act, _ = eval_sim(loss_zero, sym_zero)
      return np.concatenate([flatten_vectors(grad_act).eval().astype(np.float64),
                             flatten_vectors(grad_position).eval().astype(np.float64), 
                             flatten_vectors(grad_velocity).eval().astype(np.float64),
                             flatten_vectors(grad_accel).eval().astype(np.float64)])
      #return flatten_vectors(grad).eval().astype(np.float64)

    def get_bounds(self):
      #actuation
      lb = []
      ub = []
      acts = trainables[0]
      lb += [-1.0 / num_links] * tf.size(input=acts).eval()
      ub += [1.0 / num_links] * tf.size(input=acts).eval()
      designs = trainables[1]
      lb += [3] * tf.size(input=designs).eval()
      ub += [40] * tf.size(input=designs).eval()
  
      return (lb, ub)
      
      
  #IPython.embed()
  uda = pg.nlopt("slsqp")
  #uda = ppnf.snopt7(screen_output = False, library = "/home/aespielberg/snopt/lib/libsnopt7.so")
  algo = pg.algorithm(uda)
  #algo.extract(pg.nlopt).local_optimizer = pg.nlopt('lbfgs')
  
  algo.extract(pg.nlopt).maxeval = 50
  algo.set_verbosity(1)
  udp = RobotProblem(False)
  bounds = udp.get_bounds()
  mean = (np.array(bounds[0]) + np.array(bounds[1])) / 2.0
  num_vars = len(mean)
  prob = pg.problem(udp)
  pop = pg.population(prob, size = 1)   
  
  #TODO: initialize both parts different here
  acts = trainables[0]
  designs = trainables[1]
  
  std_act = np.ones(tf.size(input=acts).eval()) * 0.1
  std_young = np.ones(tf.size(input=designs).eval()) * 0.0
  #IPython.embed()
  std = np.concatenate([std_act, std_young])
  #act_part =  np.random.normal(scale=0.1, loc=mean, size=(tf.size(acts).eval(),))
  #young_part = 10.0 * tf.size(designs).eval()
  
  
  pop.set_x(0,np.random.normal(scale=std, loc=mean, size=(num_vars,)))
  #IPython.embed()
  
  pop.problem.c_tol = [1e-6] * prob.get_nc()
  #pop.problem.c_tol = [1e-4] * prob.get_nc()
  pop.problem.f_tol_rel = [100000.0]
  #IPython.embed()
  pop = algo.evolve(pop)
  IPython.embed()      
  
  #IPython.embed() #We need to refactor this for real
  old_x = pop.champion_x
  assert False
  udp = RobotProblem(True)
  prob = pg.problem(udp)
  pop = pg.population(prob, size = 1)   
  pop.set_x(0,old_x)
  pop.problem.c_tol = [1e-6] * prob.get_nc()
  #pop.problem.f_tol = [1e-6] 
  pop.problem.f_tol_rel = [1e-4]  
  pop = algo.evolve(pop)
   
  #now a second time
  
  
  _, _, memo = eval_sim(loss)
  sim.visualize(memo)



if __name__ == '__main__':
  sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.compat.v1.Session(config=sess_config) as sess:
    main(sess=sess)
