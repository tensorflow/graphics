import tensorflow as tf
from vector_math import *
import numpy as np
from time_integration import InitialSimulationState, UpdatedSimulationState
from time_integration import tf_precision, np_precision
from memo import Memo
import IPython
import os
from typing import Iterable, Callable, Union, List

generate_timeline = False

try:
  import ctypes
except:
  print("Warning: cannot import taichi or CUDA solver.")

def get_new_bc(res, boundary_thickness=4, boundary = None, boundary_ = None):
  if len(res) == 2:
    bc_parameter = np.zeros(
      shape=(1,) + res + (1,), dtype=np_precision)
    bc_parameter += 0.0  # Coefficient of friction
    bc_normal = np.zeros(shape=(1,) + res + (len(res),), dtype=np_precision)
    bc_normal[:, :boundary_thickness] = (1, 0)
    bc_normal[:, res[0] - boundary_thickness - 1:] = (-1, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1)
    for i in range(res[0]):
      ry = boundary(i)
      x, iy = i, int(np.round(ry))
      k = boundary_(i)
      L = (k ** 2 + 1) ** 0.5
      dx, dy = (-k / L, 1 / L)
      for y in range(iy - 5, iy + 1):
        bc_normal[:, x, y] = (dx, dy)
        bc_parameter[:, x, y] = 0
  return bc_parameter, bc_normal

def get_bounding_box_bc(res : Iterable, friction : float=0.5, boundary_thickness : int=3) -> np.ndarray:
  """
  Create a box around the entire world with specified friction and box thickness.  Useful for making sure particles don't "escape."  The resulting tuple can be passed into the simulation constructor.
  
  :param res: The resolution of the world in terms of grid cell count.
  :type res: tuple of ints.
  :param friction: The dynamic friction coefficient of the bounding box
  :type friction: float, optional
  :param boundary_thickness: The number of grid cells thickness to make non-penetrable on each side of the world.
  :type boundary_thickness: int, optional
  :return bc_parameter: The friction of the boundary condition, uniform, batch_size x res x 1.  Rank 4 or 5 total depending on rank of res (2 or 3).
  :rtype bc_paramter: np.array
  :return bc_normal: The normal of the boundary condition at each grid cell, uniform, batch_size x res x dim.  Rank 4 or 5 total depending on rank of res (2 or 3).
  :rtype bc_normal: np.array
  """
  if len(res) == 2:
    bc_parameter = np.zeros(
      shape=(1,) + res + (1,), dtype=np_precision)
    bc_parameter += friction # Coefficient of friction
    bc_normal = np.zeros(shape=(1,) + res + (len(res),), dtype=np_precision)
    # len([:boundary_thickness]) = boundary_thickness.
    # len([res[0]:]) = []
    # len([res[0] - boundary_thickness:]) = boundary_thickness.
    bc_normal[:, :boundary_thickness] = (1, 0)
    bc_normal[:, res[0] - boundary_thickness:] = (-1, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1)
    bc_normal[:, :, res[1] - boundary_thickness:] = (0, -1)
  else:
    assert len(res) == 3
    bc_parameter = np.zeros(
      shape=(1,) + res + (1,), dtype=np_precision)
    bc_parameter += friction  # Coefficient of friction
    bc_normal = np.zeros(shape=(1,) + res + (len(res),), dtype=np_precision)
    bc_normal[:, :boundary_thickness] = (1, 0, 0)
    bc_normal[:, res[0] - boundary_thickness - 1:] = (-1, 0, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1, 0)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1, 0)
    bc_normal[:, :, :boundary_thickness] = (0, 1, 0)
    bc_normal[:, :, res[1] - boundary_thickness - 1:] = (0, -1, 0)
    bc_normal[:, :, :, :boundary_thickness] = (0, 0, 1)
    bc_normal[:, :, :, res[2] - boundary_thickness - 1:] = (0, 0, -1)
  return bc_parameter, bc_normal

class Simulation:
  """The simulation class.  In order to simulate a robot, the simulator must be created, then its run and eval_gradients methods may  be called.  Once initialized, the simulation should be treated as immutable.  The simulation takes in the world deformation (boundary condition, gravity, etc.), the robot configuration, and simulator settings.  Constructor parameters:
  
  :param sess: The current tensorflow session.
  :type sess: tf.session
  :param grid_res: The grid resolution of the world, in terms of the grid node count.  Length dim.
  :type grid_res: tuple
  :param num_particles: The total number of particles to simulate.
  :type num_particles: int
  :param controller: A function which takes as input a :class:`SimulationState` and returns the actuation that state should produce (closed loop control) and returns the stress actuation matrix (batch_size x dim x dim x num_particles) to apply to each particle and optional debug information.
  :type controller: function, optional
  :param F_controller: A function which takes as input a :class:`SimulationState` and returns the actuation that state should produce (closed loop control) and returns a global force vector (batch_size x dim x num_particles) to apply to each particle and optional debug information.
  :type F_controller: function, optional
  :param gravity: an indexible type of length dim of floats, denoting the world gravity.
  :type gravity: indexible type, optional
  :param dt: The simulation timestep.
  :type dt: float, optional.
  :param dx: The (uniform) length of each grid cell (in all directions)
  :type dx: float, optional
  :param bc: The boundary condition of the world.  tuple of length 2 of form friction (batch_size, res, 1), normals (batch_size, res, dim).
  :type bc: tuple, optional.
  :param E: The Young's modulus of the particles.  Can either be a scalar (uniform stiffness) or a tensor (spatially varying) of size batch_size x 1 x num_particles.
  :type E: float or tf.tensor, optional
  :param nu: The Poisson's ratio of the particles.  Can either be a scalar (uniform stiffness) or a tensor (spatially varying) of size batch_size x 1 x num_particles.
  :type nu: float or tf.tensor, optional
  :param m_p: The mass of the particles.  Can either be a scalar (uniform stiffness) or a tensor (spatially varying) of size batch_size x 1 x num_particles.
  :type m_p: float or tf.tensor, optional
  :param V_p: The volume of the particles.  Can either be a scalar (uniform stiffness) or a tensor (spatially varying) of size batch_size x 1 x num_particles.
  :type V_p: float or tf.tensor, optional
  :param batch_size: The batch size.
  :type batch_size: int, optional
  :param scale: An amount to scale the visualization by.
  :type scale: int, optional
  :param damping: A Rayleigh damping parameter.
  :type damping: float, optional
  :param part_size: Number of steps to simulate sequentially on the GPU before returning any results to the CPU.
  :type part_size:  int, optional
  :param use_visualize: Should we store data for visualization purposes?
  :type use_visualize: bool, optional
  :param use_cuda: Simulate on GPU?  If not, CPU
  :type use_cuda: bool, optional
  :param substeps: Number of steps to simulate sequentially on the GPU before returning any results to the CPU (for recording the memo or applying controller input),  This is similar to part_size, but will not call the controller while applying substeps.
  :type substeps: int, optional
  """

  def __init__(self,
               sess : tf.compat.v1.Session,
               # Let's use grid_res to represent the number of grid nodes along each direction.
               # So grid_res = (3, 3) means 2 x 2 voxels.
               grid_res : Iterable,
               num_particles : int,
               controller : Callable[['SimulationState'], tf.Tensor]=None,
               F_controller : Callable[['SimulationState'], tf.Tensor]=None,
               gravity : Iterable=(0, -9.8),
               dt : float=0.01,
               dx : float=None,
               bc : np.ndarray=None,
               E : Union[float, tf.Tensor]=10,  # E can be either a scalar or a tensor.
               nu : Union[float, tf.Tensor]=0.3,
               m_p : Union[float, tf.Tensor]=1, # m_p can be either a scalar or a tensor.
               V_p : Union[float, tf.Tensor]=1,
               batch_size : int=1,
               scale : int=None,
               damping : float=0,
               part_size : int=1,
               use_visualize : bool=True,
               use_cuda : bool=True,
               substeps : int=1,
               use_neohookean=False,
               incompressibility_exp=-1.0) -> None:
    # Set up the constraints for the dynamic mode:
    self.substeps = substeps
    self.use_cuda = use_cuda
    self.use_neohookean = use_neohookean
    self.incompressibility_exp = incompressibility_exp
               
    self.dim = len(grid_res)
    self.InitialSimulationState = InitialSimulationState
    self.UpdatedSimulationState = UpdatedSimulationState
    if self.dim == 2:
      self.identity_matrix = identity_matrix
    else:
      self.identity_matrix = identity_matrix_3d

    assert batch_size == 1, "Only batch_size = 1 is supported."

    self.sess = sess
    self.num_particles = num_particles
    if scale is None:
      self.scale = 900 // grid_res[0]
    else:
      self.scale = scale
    if (self.scale * grid_res[0]) % 2 != 0 or (self.scale * grid_res[1]) % 2 != 0:
      self.scale += 1 #keep it even
      
    self.grid_res = grid_res
    self.dim = len(self.grid_res)
    if dx is None:
      # grid_res[0] - 1 because grid_res is the number of nodes, not cells.
      # This way, Node 0 is at 0 and Node[grid_res[0] - 1] is at 1.0.
      # We have grid_res[0] - 1 cells that fully covers [0, 1].
      dx = 1.0 / (grid_res[0] - 1)
    self.batch_size = batch_size
    self.damping = damping

    if bc is None and self.dim == 2:
      bc = get_bounding_box_bc(grid_res)
      
    if bc is not None:
      self.bc_parameter, self.bc_normal = bc
    self.initial_state = self.InitialSimulationState(self, controller, F_controller)
    self.grad_state = self.InitialSimulationState(self, controller, F_controller)
    self.gravity = gravity
    self.dx = dx
    self.dt = dt
    if type(E) in [int, float]:
      self.youngs_modulus = np.full((batch_size, 1, num_particles), E)
    else:
      self.youngs_modulus = E
    assert self.youngs_modulus.shape == (batch_size, 1, num_particles)
    # nu has to be a scalar.
    if type(nu) in [int, float]:
      self.nu = np.full((batch_size, 1, num_particles), nu)
    else:
      self.nu = nu
    assert self.nu.shape == (batch_size, 1, num_particles)
    # We allow m_p to be a scalar or a Tensor.
    if type(m_p) in [int, float]:
      self.m_p = np.full((batch_size, 1, num_particles), m_p)
    else:
      self.m_p = m_p
    assert self.m_p.shape == (batch_size, 1, num_particles)
    # V_p has to be a scalar
    if type(V_p) not in [int, float]:
      print("Error: V_p must be a scalar.")
      raise NotImplementedError
    self.V_p = V_p
    self.inv_dx = 1.0 / dx

    self.part_size = part_size
    self.states = [self.initial_state]
    for i in range(part_size):
        self.states.append(self.UpdatedSimulationState(self, previous_state = self.states[-1], controller = controller, F_controller = F_controller))

    self.updated_state = self.states[-1]
    self.controller = controller
    self.parameterized_initial_state = None
    self.point_visualization = []
    self.vector_visualization = []
    self.frame_counter = 0
    self.use_visualize = use_visualize

  def stepwise_sym(self, expr):
    temp = tf.stack([expr(state) for state in self.states[1:]], axis = 0)
    return tf.reduce_sum(input_tensor=temp, axis = 0)

  def visualize_2d(self, memo : 'Memo', interval : int=1, batch : int=0, export : 'Export'=None, show : bool=False, folder : str=None, youngs_to_color : Callable[[float], float]=None) -> None:
    """
    Visualize a simulation in 2D.  Uses a built-in visualizer using OpenCV.
    
    :param memo: The memo returned by the simulation to visualize
    :type memo: :class:`Memo`    
    :param interval: The number of steps to skip in visualization between frames.
    :type interval: int, optional
    :param batch: The batch num
    :type batch: int, optional
    :param export: An export object providing information for saving the visualization.  Set as None if saving is not desired.
    :type export: :class:`Export`, optional
    :param show: Whether or not we should show the visualization to the screen
    :type show: bool, optional
    :param folder: A directory of where to save the visualization
    :type folder: str, optional
    :param youngs_to_color: A float to float mapping from Youngs modulus to between 0 and 1 for the visualizing Young's modulus in the blue color channel.  Set to None if visualizing youngs modulus is not desired.
    :type youngs_to_color: func, optional
    """
    import math
    import cv2
    scale = self.scale

    b = batch
    # Pure-white background
    background = 0.5 * np.ones(
      (self.grid_res[0], self.grid_res[1], 3), dtype=np_precision)
    background[:,:,2]=1.0

    for i in range(self.grid_res[0]):
      for j in range(self.grid_res[1]):
        if self.bc_parameter[0][i][j] == -1:
          background[i][j][0] = 0
        normal = self.bc_normal[0][i][j]
        if np.linalg.norm(normal) != 0:
          background[i][j] *= 0.7
    background = cv2.resize(
      background, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    alpha = 0.50
    last_image = background

    if folder:
      os.makedirs(folder, exist_ok=True)

      
    for i, (s, act, points, vectors) in enumerate(zip(memo.steps, memo.actuations, memo.point_visualization, memo.vector_visualization)):
      if i % interval != 0:
        continue

      particles = []
      pos = s[0][b] * self.inv_dx + 0.5
      pos = np.transpose(pos)      
      youngs = np.ndarray.flatten(s[6][b])

      scale = self.scale

      img = background.copy()      
      for j, (young, p) in enumerate(zip(youngs, pos)):
        x, y = tuple(map(lambda t: math.floor(t * scale), p))
        if youngs_to_color is None:
          intensity = 0.2
        else:
          intensity = youngs_to_color(young)        
        if act is not None:
          a = act[0, :, :, j]
          max_act = 2.0 #TODO: pass this in.  Right now overriding the max.
        else:
          a = np.array([[0, 0], [0, 0]])
          max_act = 1.0        
        
        red = a[0, 0] / max_act/ 2.0 + 0.5
        green = a[1, 1] / max_act / 2.0 + 0.5
        #red = np.sqrt(a[0, 0]**2 + a[1, 1]**2) / max_act / 2.0
        #green = 0.5
        color = (red, green, intensity)
        cv2.circle(img, (y, x), radius=3, color=color, thickness=-1)        
        particles.append((p[0], p[1]) + (young, color[1], color[2], a[0][0], a[0][1], a[1][0], a[1][1]))

      dots = []
      for dot in points:
        coord, color, radius = dot
        #handle a whole bunch of points here:
        for pt in coord:
          pt = np.int32((pt * self.inv_dx + 0.5) * scale)
          cv2.circle(img, (pt[1], pt[0]), color=color, radius=radius, thickness=-1)
          dots.append(tuple(pt) + tuple(color))

      for line in vectors:
        pos, vec, color, gamma = line
        pos = (pos * self.inv_dx + 0.5) * scale
        vec = vec * gamma + pos
        cv2.line(img, (pos[b][1], pos[b][0]), (vec[b][1], vec[b][0]), color = color, thickness = 1)

      last_image = 1 - (1 - last_image) * (1 - alpha)
      last_image = np.minimum(last_image, img)
      img = last_image.copy()
      img = img.swapaxes(0, 1)[::-1, :, ::-1]

      if show:
        cv2.imshow('Differentiable MPM Simulator', img)
        cv2.waitKey(1)
      if export is not None:
        export(img)

      if folder:
        with open(os.path.join(folder, 'frame{:05d}.txt'.format(i)), 'w') as f:
          for p in particles:
            print('part ', end=' ', file=f)
            for x in p:
              print(x, end=' ', file=f)
            print(file=f)
          for d in dots:
            print('vis ', end=' ', file=f)
            for x in d:
              print(x, end=' ', file=f)
            print(file=f)

    if export is not None:
      export.wait()

  def visualize_3d(self, memo, interval=1, batch=0, export=None, show=False, folder=None):
    if export:
      frame_count_delta = self.frame_counter
    else:
      frame_count_delta = 0
    print(folder)
    print("Warning: skipping the 0th frame..")
    for i, (s, act, points, vectors) in enumerate(zip(memo.steps, memo.actuations, memo.point_visualization, memo.vector_visualization)):
      if i % interval != 0 or i == 0:
        continue
      pos = s[0][batch].copy()
      if output_bgeo:
        task = Task('write_partio_c')
        #print(np.mean(pos, axis=(0)))
        ptr = pos.ctypes.data_as(ctypes.c_void_p).value
        suffix = 'bgeo'
      else:
        task = Task('write_tcb_c')
        #print(np.mean(pos, axis=(0)))
        act = np.mean(act, axis=(1, 2), keepdims=True)[0, :, 0] * 9
        pos = np.concatenate([pos, act], axis=0).copy()
        ptr = pos.ctypes.data_as(ctypes.c_void_p).value
        suffix = 'tcb'
      if folder is not None:
        os.makedirs(folder, exist_ok=True)
      else:
        folder = '.'
      task.run(str(self.num_particles),
               str(ptr), '{}/{:04d}.{}'.format(folder, i // interval + frame_count_delta, suffix))
      self.frame_counter += 1

  def visualize(self, memo, interval=1, batch=0, export=None, show=False, folder=None):
    if self.dim == 2:
      self.visualize_2d(memo, interval, batch, export, show, folder)
    else:
      self.visualize_3d(memo, interval, batch, export, show, folder)


  def initial_state_place_holder(self):
    return self.initial_state.to_tuple()

  def evaluate_points(self, state, extra={}):
    if self.point_visualization is None:
      return []
    pos_tensors = [p[0] for p in self.point_visualization]
    feed_dict = {self.initial_state.to_tuple(): state}
    feed_dict.update(extra)

    pos = self.sess.run(pos_tensors, feed_dict=feed_dict)
    return [(p,) + tuple(list(r)[1:])
            for p, r in zip(pos, self.point_visualization)]

  def evaluate_vectors(self, state, extra = {}):
    if self.vector_visualization is None:
      return []
    pos_tensors = [v[0] for v in self.vector_visualization]
    vec_tensors = [v[1] for v in self.vector_visualization]
    feed_dict = {self.initial_state.to_tuple(): state}
    feed_dict.update(extra)
    pos = self.sess.run(pos_tensors, feed_dict=feed_dict)
    vec = self.sess.run(vec_tensors, feed_dict=feed_dict)
    return [(p,v) + tuple(list(r)[2:]) for p, v, r in zip(pos, vec, self.vector_visualization)]

  def run(self,
          num_steps : int,
          initial_state : 'SimulationState'=None,
          initial_feed_dict : dict={},
          iteration_feed_dict : dict={},
          loss : Callable[['SimulationState'], tf.Tensor]=None,
          stepwise_loss=None,
          skip_states : bool=True,
          freq : int=1) -> 'Memo':
          
    """Runs forward simulation for a specified numer of time steps from a specified initial state.
    
    :param num_steps: The number of steps to simulate forward for.
    :type num_steps: int
    :param initial_feed_dict: An optional tf.session feed dictionary mapping iniital state placeholder tensors to values to be evaluated during simulation.  Useful if parts of the simulation should be represented parametrically.
    :type initial_feed_dict: dict, optional
    :param iteration_feed_dict:  An optional tf.session feed dictionary mapping non-initial state placeholder tensors to values to be evaluated during simulation.  Useful if parts of the simulation should be represented parametrically.
    :type iteration_feed_dict: dict, optional
    :param loss: A loss function that takes in a :class:`SimulationState` and returns a tensor scalar loss.  Gradients will be computed with respect to this loss.
    :type loss: func, optional
    :return: A memo for the simulation.
    :rtype: :class:`Memo`
    """
    #DOCTODO: other params
    
    assert num_steps % self.part_size == 0
    if callable(iteration_feed_dict) and stepwise_loss is not None and not skip_states:
      print('ERROR: this case is not implemented yet.')
      sys.exit(-1)
    memo = Memo()
    memo.initial_feed_dict = initial_feed_dict
    iter_feed_dict_eval = iteration_feed_dict
    if callable(iteration_feed_dict):
      iter_feed_dict_eval = iteration_feed_dict(0)
    memo.iteration_feed_dict = iteration_feed_dict
    if initial_state is None:
      initial_state = self.initial_state
    memo.initial_state = initial_state

    initial_evaluated = []
    for t in initial_state:
      if isinstance(t, tf.Tensor):
        initial_evaluated.append(self.sess.run(t, initial_feed_dict))
      else:
        initial_evaluated.append(t)

    memo.steps = [initial_evaluated]
    memo.actuations = [None]
    if self.use_visualize:
      memo.point_visualization.append(self.evaluate_points(memo.steps[0], iter_feed_dict_eval))
      memo.vector_visualization.append(self.evaluate_vectors(memo.steps[0], iter_feed_dict_eval))
    rest_steps = num_steps
    t = 0
    idx = 0
    while rest_steps > 0:      
      now_step = min(rest_steps, self.part_size)
      memo.last_step = now_step
      rest_steps -= now_step
      feed_dict = {self.initial_state.to_tuple(): memo.steps[-1]}
      feed_dict.update(iter_feed_dict_eval)
      # Update time and iter_feed_dict_eval.
      t += self.dt * self.part_size
      if callable(iteration_feed_dict):
        iter_feed_dict_eval = iteration_feed_dict(t)

      if generate_timeline:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        timeline_options = {
          'options': options,
          'run_metadata': run_metadata
        }
      else:
        timeline_options = {}

      has_act = self.updated_state.controller is not None
      
      if stepwise_loss is None:
        ret_ph = [self.states[now_step].to_tuple()]
        if has_act:
          ret_ph += [self.states[now_step].actuation]
        ret = self.sess.run(ret_ph, feed_dict=feed_dict, **timeline_options)
        '''ANDY DEBUG'''
        '''
        with tf.variable_scope("globals", reuse=tf.AUTO_REUSE):
          first_time = tf.get_variable("first_time", initializer = tf.constant(1.0), dtype=tf.float32)
          first_time = self.sess.run(first_time)
        '''
        if False:     #Uncomment me for debug info                 
          tmp = self.sess.run("image_test:0", feed_dict=feed_dict, **timeline_options)   
          recon_scalar = self.sess.run("recon_diff/Squeeze:0", feed_dict=feed_dict, **timeline_options)
          recon_image = self.sess.run("recon_image:0", feed_dict=feed_dict, **timeline_options)
          feed_dict[self.run_image] = tmp
          feed_dict[self.recon_error] = recon_scalar
          feed_dict[self.recon_image] = recon_image
          summary = self.merged_summary_op.eval(feed_dict=feed_dict)
          self.summary_writer.add_summary(summary, self.summary_iter)
          self.summary_iter += 1
          inp = self.sess.run("inputs:0", feed_dict=feed_dict, **timeline_options)
          f=open('pca.txt','ab')
          #print('write')
          np.savetxt(f,inp[0, :, 0])
          f.flush()
          '''
          import matplotlib.pyplot as plt
          plt.imshow(tmp[0])
          plt.show()
          '''
        '''END ANDY DEBUG'''
        
        memo.steps.append(ret[0])
        if has_act:
          memo.actuations.append(ret[1])
      else:
        if skip_states:
          ret_ph = [self.states[now_step].to_tuple()]
          if has_act:
            ret_ph += [self.states[now_step].actuation]
          ret = self.sess.run(ret_ph, feed_dict=feed_dict, **timeline_options)
          s = [ret[0]]
          a = [ret[1] if has_act else None]
          # Compute the loss.
          loss_feed_dict = { self.initial_state.to_tuple(): s }
          loss_feed_dict.update(iter_feed_dict_eval)
          swl = self.sess.run(loss, feed_dict=loss_feed_dict, **timeline_options)
        else:
          ret_states = [s.to_tuple() for s in self.states]
          s, swl = self.sess.run((ret_states[1:], stepwise_loss),
                                    feed_dict=feed_dict, **timeline_options)
          if has_act:
            ret_actuations = [s.actuation for s in self.states]
            a = self.sess.run(ret_actuations[1:],
                              feed_dict=feed_dict, **timeline_options)
          else:
            a = [None] * len(s)
        memo.steps += s
        memo.actuations += a
        # It seems that update_stepwise_loss(swl) did not work when swl is a scalar.
        memo.update_stepwise_loss(swl)

      if self.use_visualize:
        if stepwise_loss is None or skip_states:
          memo.point_visualization.append(self.evaluate_points(memo.steps[-1], iter_feed_dict_eval))
          memo.vector_visualization.append(self.evaluate_vectors(memo.steps[-1], iter_feed_dict_eval))
        else:
          for s in memo.steps[-now_step:]:
            memo.point_visualization.append(self.evaluate_points(s, iter_feed_dict_eval))
            memo.vector_visualization.append(self.evaluate_vectors(s, iter_feed_dict_eval))

      # Create the Timeline object, and write it to a json file
      if generate_timeline:
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
          f.write(chrome_trace)
        IPython.embed()

    if loss is not None and stepwise_loss is None:
      feed_dict = {self.initial_state.to_tuple(): memo.steps[-1]}
      feed_dict.update(iter_feed_dict_eval)
      memo.loss = self.sess.run(loss, feed_dict=feed_dict)
    
    return memo

  @staticmethod
  def replace_none_with_zero(grads, data):
    ret = []
    for g, t in zip(grads, data):
      if g is None:
        ret.append(tf.zeros_like(t))
      else:
        ret.append(g)
    return tuple(ret)

  def set_initial_state(self, initial_state):
    self.parameterized_initial_state = initial_state

  '''
  def gradients_step_sym(self, loss, variables, steps):
    step_grad_variables = tf.gradients(
        ys = self.states[steps].to_tuple(),
        xs = self.initial_state.to_tuple(),
        grad_ys = self.grad_state.to_tuple())

    step_grad_variables = self.replace_none_with_zero(step_grad_variables,
                                                      variables)

    step_grad_states = tf.gradients(
        ys=self.states[steps].to_tuple(),
        xs=self.initial_state.to_tuple(),
        grad_ys=self.grad_state.to_tuple())

    step_grad_states = self.replace_none_with_zero(
        step_grad_states, self.initial_state.to_tuple())

    return {'steps_grad_variables': }
  '''

  def gradients_sym(self, loss : Callable[[float], float], variables : Iterable[tf.Variable], use_stepwise_loss : bool=False, skip_states : bool=True) -> dict:
    """
    Computes a sym dictionary with all the information needed to efficiently compute gradients.  This must be called once for a problem and the resulting dictionary must be pased in to any eval_gradients calls.
    
    :param loss: A loss function that takes in a :class:`SimulationState` and returns a tensor scalar loss.  Gradients will be computed with respect to this loss.
    :type loss: func, optional
    :param variables: A collection of tf.Variables with respect to which gradients are computed.
    :type variables: collection
    :return: A dict to be passed into eval_gradients.
    :rtype: :class:`dict`
    
    """
    # loss = loss(initial_state)
    variables = tuple(variables)

    last_grad_sym = tf.gradients(ys=loss, xs=self.initial_state.to_tuple())

    last_grad_sym_valid = self.replace_none_with_zero(
        last_grad_sym, self.initial_state.to_tuple())

    for v in variables:
      assert tf.convert_to_tensor(v).dtype == tf_precision, v

    # partial S / partial var
    step_grad_variables = tf.gradients(
        ys=self.updated_state.to_tuple(),
        xs=variables,
        grad_ys=self.grad_state.to_tuple())

    step_grad_variables = self.replace_none_with_zero(step_grad_variables,
                                                      variables)

    # partial S / partial S'
    if use_stepwise_loss and not skip_states:
      updated_state = self.states[1]
    else:
      updated_state = self.updated_state
    step_grad_states = tf.gradients(
      ys=updated_state.to_tuple(),
      xs=self.initial_state.to_tuple(),
      grad_ys=self.grad_state.to_tuple())

    step_grad_states = self.replace_none_with_zero(
        step_grad_states, self.initial_state.to_tuple())

    parameterized_initial_state = tuple([
        v for v in self.parameterized_initial_state if isinstance(v, tf.Tensor)
    ])
    parameterized_initial_state_indices = [
        i for i, v in enumerate(self.parameterized_initial_state)
        if isinstance(v, tf.Tensor)
    ]

    def pick(l):
      return tuple(l[i] for i in parameterized_initial_state_indices)

    initial_grad_sym = tf.gradients(
        ys=parameterized_initial_state,
        xs=variables,
        grad_ys=pick(self.grad_state.to_tuple()))

    initial_grad_sym_valid = self.replace_none_with_zero(
        initial_grad_sym, variables)

    sym = {}
    sym['last_grad_sym_valid'] = last_grad_sym_valid
    sym['initial_grad_sym_valid'] = initial_grad_sym_valid
    sym['step_grad_variables'] = step_grad_variables
    sym['step_grad_states'] = step_grad_states
    sym['parameterized_initial_state'] = parameterized_initial_state
    sym['pick'] = pick
    sym['variables'] = variables

    return sym
    
  def eval_gradients(self, sym : dict, memo : 'Memo', use_stepwise_loss : bool=False, skip_states : bool=True) -> List[np.ndarray]:
    """Compute the gradients of a completed simulation.
    :param sym: The symbolic dictionary returned by :meth:`gradients_sym`
    :type sym: dict
    :param memo: The memo returned by a simulation from :meth:`run`
    :type memo: :class:`Memo`
    :return: A collection of np.array with the same structure as the variables passed into :meth:`gradient_sym`; a tuple of np.arrays
    :rtype: tuple
    """
    if use_stepwise_loss and not skip_states:
      print('ERROR: this case is not implemented yet.')
      sys.exit(0)
    last_grad_sym_valid = sym['last_grad_sym_valid']
    initial_grad_sym_valid = sym['initial_grad_sym_valid']
    step_grad_variables = sym['step_grad_variables']
    step_grad_states = sym['step_grad_states']
    parameterized_initial_state = sym['parameterized_initial_state']
    pick = sym['pick']
    variables = sym['variables']

    grad = [np.zeros(shape=v.shape, dtype=np_precision) for v in variables]
    feed_dict = {self.initial_state.to_tuple(): memo.steps[-1]}
    t = self.dt * self.part_size * (len(memo.steps) - 1)
    iter_feed_dict_eval = memo.iteration_feed_dict
    if callable(memo.iteration_feed_dict):
      iter_feed_dict_eval = memo.iteration_feed_dict(t)
    feed_dict.update(iter_feed_dict_eval)
    last_grad_valid = self.sess.run(last_grad_sym_valid, feed_dict=feed_dict)


    last_step_flag = memo.last_step != self.part_size
    if last_step_flag:
      raise Exception("Unfinished step")

    if use_stepwise_loss and not skip_states:
      updated_state = self.states[1]
    else:
      updated_state = self.updated_state

    for i in reversed(range(1, len(memo.steps))):
      if any(v is not None for v in step_grad_variables):
        feed_dict = {
          self.initial_state.to_tuple(): memo.steps[i - 1],
          updated_state.to_tuple(): memo.steps[i],
          self.grad_state.to_tuple(): last_grad_valid
        }
        feed_dict.update(iter_feed_dict_eval)
        grad_acc = self.sess.run(step_grad_variables, feed_dict=feed_dict)
        for g, a in zip(grad, grad_acc):
          g += a

      feed_dict = {
        self.initial_state.to_tuple(): memo.steps[i - 1],
        updated_state.to_tuple(): memo.steps[i],
        self.grad_state.to_tuple(): last_grad_valid
      }
      feed_dict.update(iter_feed_dict_eval)
      last_grad_valid = self.sess.run(step_grad_states, feed_dict=feed_dict)
      # Update iter_feed_dict_eval.
      t -= self.dt * self.part_size
      if callable(memo.iteration_feed_dict):
        iter_feed_dict_eval = memo.iteration_feed_dict(t)
      # We define stepwise loss on memo.steps[1:], not the whole memo.steps.
      if use_stepwise_loss and i != 1:
        feed_dict = {self.initial_state.to_tuple(): memo.steps[i - 1]}
        feed_dict.update(iter_feed_dict_eval)
        last_step_grad_valid = self.sess.run(last_grad_sym_valid, feed_dict=feed_dict)
        for g, a in zip(last_grad_valid, last_step_grad_valid):
          g += a

    assert np.isclose(t, 0)

    if any(v is not None for v in initial_grad_sym_valid):
      feed_dict = {}
      feed_dict[parameterized_initial_state] = pick(memo.steps[0])
      feed_dict[pick(self.grad_state.to_tuple())] = pick(last_grad_valid)
      grad_acc = self.sess.run(initial_grad_sym_valid, feed_dict=feed_dict)
      for g, a in zip(grad, grad_acc):
        g += a

    return grad

  def get_initial_state(self,
                        position : Union[tf.Tensor, np.ndarray],
                        velocity : Union[tf.Tensor, np.ndarray]=None,
                        particle_mass : Union[tf.Tensor, np.ndarray]=None,
                        particle_volume : Union[tf.Tensor, np.ndarray]=None,
                        youngs_modulus : Union[tf.Tensor, np.ndarray]=None,
                        poissons_ratio : Union[tf.Tensor, np.ndarray]=None,
                        deformation_gradient : Union[tf.Tensor, np.ndarray]=None) -> 'SimulationSate':
    
    """Gets an :class:`InitialSimulationState` object to pass into sim.run's initial state.  Defines the initial state of a simulation.  Users may specify the configuration; initial particle position must at least be provided.  Defaults to simulation constructor values.
    
    :param position: batch_size x dim x num_particles representing the initial particle positions.
    :type position: np.array or tf.tensor
    :param velocity: batch_size x dim x num_particles represneting the initial particle velocities.
    :type velocity: np.array or tf.tensor
    :param velocity: batch_size x 1 x num_particles represneting the initial particle young's moduluses.
    :type velocity: np.array or tf.tensor
    :param velocity: batch_size x 1 x num_particles represneting the initial poissons' ratios.
    :type velocity: np.array or tf.tensor
    :param velocity: batch_size x 1 x num_particles represneting the initial particle masses.
    :type velocity: np.array or tf.tensor
    :param velocity: batch_size x 1 x num_particles represneting the initial particle volumes.
    :type velocity: np.array or tf.tensor
    :param velocity: batch_size x dim x dim x num_particles represneting the initial particle deformation_gradient.
    :type velocity: np.array or tf.tensor
    :return: :class:`InitialSimulationState` representing the initial state of the MPM system.  Most useful for passing into :meth:`run`
    :rtype: :class:`InitialSimulationState`
    """
    acceleration = np.zeros(
      shape=[self.batch_size, self.dim, self.num_particles], dtype = np_precision)
    if velocity is not None:
      initial_velocity = velocity
    else:
      initial_velocity = np.zeros(
          shape=[self.batch_size, self.dim, self.num_particles], dtype = np_precision)
    if deformation_gradient is None:
      deformation_gradient = self.identity_matrix +\
                             np.zeros(shape=(self.batch_size, 1, 1, self.num_particles), dtype = np_precision),
    affine = self.identity_matrix * 0 + \
                           np.zeros(shape=(self.batch_size, 1, 1, self.num_particles), dtype = np_precision),
    batch_size = self.batch_size
    num_particles = self.num_particles

    if type(particle_mass) in [int, float]:
      self.m_p = np.ones(shape=(batch_size, 1, num_particles), dtype=np_precision) * particle_mass
    elif particle_mass is not None:
      self.m_p = particle_mass
    assert self.m_p.shape == (batch_size, 1, num_particles)
    particle_mass = self.m_p

    # For now, we require particle_Volume to be scalars.
    if type(particle_volume) in [int, float]:
      self.V_p = particle_volume
    elif particle_volume is not None:
      print("ERROR: particle_volume has to be a scalar.")
      raise NotImplementedError
    particle_volume = np.ones(shape=(batch_size, 1, num_particles), dtype = np_precision) * self.V_p

    if type(youngs_modulus) in [int, float]:
      self.youngs_modulus = np.ones(shape=(batch_size, 1, num_particles), dtype = np_precision) * youngs_modulus
    elif youngs_modulus is not None:
      self.youngs_modulus = youngs_modulus
    else:
      self.youngs_modulus = np.ones(shape=(batch_size, 1, num_particles), dtype = np_precision) * self.youngs_modulus
    assert self.youngs_modulus.shape == (batch_size, 1, num_particles)
    youngs_modulus = self.youngs_modulus

    if type(poissons_ratio) in [int, float]:
      self.nu = np.ones(shape=(batch_size, 1, num_particles), dtype = np_precision) * poissons_ratio
    elif poissons_ratio is not None:
      self.nu = poissons_ratio
    else:
      self.nu = np.ones(shape=(batch_size, 1, num_particles), dtype = np_precision) * self.nu
    assert self.nu.shape == (batch_size, 1, num_particles)
    poissons_ratio = self.nu
    
    #TODO: should this be more general? 
    #IPython.embed()
    grid_mass = np.zeros(
      shape=[self.batch_size, *self.grid_res, 1], dtype = np_precision)   
    grid_velocity = np.zeros(
      shape=[self.batch_size, *self.grid_res, self.dim], dtype = np_precision)
      



    
    
    return (position, initial_velocity, deformation_gradient, affine,
            particle_mass, particle_volume, youngs_modulus, poissons_ratio, 
            0, acceleration, grid_mass, grid_velocity)

  def add_point_visualization(self, pos, color=(1, 0, 0), radius=3):
    self.point_visualization.append((pos, color, radius))
  
  def add_vector_visualization(self, pos, vector, color=(1, 0, 0), scale=10):
    self.vector_visualization.append((pos, vector, color, scale))
