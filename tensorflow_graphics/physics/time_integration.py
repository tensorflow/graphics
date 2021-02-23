import numpy as np
import tensorflow as tf
from vector_math import *
from typing import List, Callable


use_cuda = False
use_apic = True

if use_float64:
  use_cuda = False

try:
  import mpm3d
  cuda_imported = True
except:
  cuda_imported = False
if not cuda_imported:
  print("***Warning: NOT using CUDA")

kernel_size = 3

class SimulationState:
  """An object which represents the state of a single step of the simulation.  Used both to specify an initial state for running simulation and also for defining a loss on the final state.  ChainQueen handles most of the work of taking tensorflow tensors from SimulationStates and manually chaining together gradients from kernel calls for efficiency reasons.  Most of this work happens behind the scenes from the users.  SimulationState provides a number of useful members that can be helpful for defining a tensor-based loss function.  In general, most members should be thought of as being tensors that are functions of the initial state of a simulation. SimulationStates have two subclasses, class:`InitialSimulationState` for the initial state only (which initializes a lot of the problem) and class:`UpdatedSimulationState` which denotes any other simulation state.  Each SimulationState keeps track of the sim itself.
  
  :param sim: The simulation that will be used to populate these states.
  :type sim: class:`Simulation`  
  :param position: batch_size x dim x num_particles, the position of the particles in the system.
  :type position: np.array or tf.tensor
  :param velocity: batch_size x dim x num_particles, the velocity of the particles in the system.
  :type velocity: np.array or tf.tensor
  :param deformation_gradient: batch_size x dim x dim num_particles, the deformation gradient of the particles in the system.
  :type deformation_gradient: np.array or tf.tensor
  :param affine: batch_size x dim x dim num_particles, the affine matrix of the particles in the system.
  :type affine: np.array or tf.tensor
  :param particle_mass: batch_size x 1 x num_particles, the mass of every particle.
  :type particle_mass: np.array or tf.tensor
  :param particle_volume: batch_size x 1 x num_particles, the volume of each particle.
  :type particle_volume: np.array or tf.tensor
  :param youngs_modulus: batch_size x 1 x num_particles, the youngs modulus of each particle.
  :type youngs_modulus: np.array or tf.tensor
  :param poissons_ratio: batch_size x 1 x num_particles, the poissons ratio of each particle.
  :type poissons_ratio: np.array or tf.tensor
  :param step_count: The step number this simulation state corresponds to.  Can be useful for indexing into actuation sequences of an open-loop controller.
  :type step_count: int
  :param acceleration: batch_size x dim x num_particles, the acceleration of the particles in the system.
  :type acceleration: np.array or tf.tensor
  :param grid_mass: batch_size x res x 1, the rasterized mass of each grid cell
  :type grid_mas: np.array or tf.tensor
  :param grid_velocity: batch_size x res x dim, the rasterized gelocity of each grid cell
  :type grid_velocity: np.array or tf.tensor
  :param strain: batch_size x dim x dim x num_particles, the green strain matrix for each particle
  :type strain: np.array or tf.tensor
  :param d_strain_dt: batch_size x dim x dim x num_particles, the finite difference strain per time
  :type d_strain_dt: np.array or tf.tensor
  :param pressure: batch_size x dim x dim x num_particles, the pressure matrix for each particle (piola stress plus actuation contributions)
  :type pressure: np.array or tf.tensor
  :param d_pressure_dt: batch_size x dim x dim x num_particles, the finite difference pressure per time
  :type d_pressure_dt: np.array or tf.tensor
  """

  def __init__(self, sim : 'Simulation') -> None:
    self.sim = sim
    self.dim = sim.dim
    dim = self.dim
    self.grid_shape = (self.sim.batch_size,) + self.sim.grid_res + (1,)
    self.affine = tf.zeros(shape=(self.sim.batch_size, dim, dim, sim.num_particles), dtype=tf_precision)
    self.acceleration = tf.zeros(shape=(self.sim.batch_size, dim, sim.num_particles), dtype=tf_precision)
    self.position = None
    self.particle_mass = None
    self.particle_volume = None
    self.youngs_modulus = None
    self.poissons_ratio = None
    self.velocity = None
    self.deformation_gradient = None
    self.controller_states = None
    self.grid_mass = None
    self.grid_velocity = None
    self.step_count = None
    self.kernels = None
    self.debug = None
    self.actuation = None
    self.F = None
    self.use_cuda = sim.use_cuda
    self.use_neohookean = sim.use_neohookean
    self.incompressibility_exp = sim.incompressibility_exp
    if not cuda_imported:
      self.use_cuda = False
    if not self.use_cuda:
      print("***Warning: NOT using CUDA")

  def center_of_mass(self, left = None, right = None):
    return tf.reduce_sum(input_tensor=self.position[:, :, left:right] * self.particle_mass[:, :, left:right], axis=2) *\
           (1 / tf.reduce_sum(input_tensor=self.particle_mass[:, :, left:right], axis=2))

  def get_state_names(self) -> List[str]:
    """
    An ordered list of the returned data of a simulation class:`Memo` at each time step.  Here we document what each of these fields are. 
    
    :return: A list of strings of all the field names that are in the memo, in order.
    :rtype: list
    
    """
    return [
        'position', 'velocity', 'deformation_gradient', 'affine',
        'particle_mass', 'particle_volume', 'youngs_modulus', 'poissons_ratio', 'step_count', 'acceleration',
        'grid_mass', 'grid_velocity',
    ]
    
  def get_state_component_id(self, name : str) -> int:
    """gets the index of a field name in the :class:`Memo`
    
    :param name: The field name
    :type name: str
    :return: The index of the name in the Memo.
    :rtype: int
    """
    assert name in self.get_state_names()
    return self.get_state_names().index(name)

  def get_evaluated(self):
    # # batch, particle, dimension
    # assert len(self.position.shape) == 3
    # assert len(self.position.shape) == 3
    # # batch, particle, matrix dimension1, matrix dimension2
    # assert len(self.deformation_gradient.shape) == 4
    # # batch, x, y, dimension
    # assert len(self.grid_mass.shape) == 4
    # assert len(self.grid_velocity.shape) == 4

    ret = {
        'affine': self.affine,
        'position': self.position,
        'velocity': self.velocity,
        'deformation_gradient': self.deformation_gradient,
        'acceleration': self.acceleration,
        'controller_states': self.controller_states,
        'grid_mass': self.grid_mass,
        'grid_velocity': self.grid_velocity,
        'kernels': self.kernels,
        'particle_mass': self.particle_mass,
        'particle_volume': self.particle_volume,
        'youngs_modulus': self.youngs_modulus,
        'poissons_ratio': self.poissons_ratio,
        'step_count': self.step_count,
        'debug': self.debug,
    }
    ret_filtered = {}
    for k, v in ret.items():
      if v is not None:
        ret_filtered[k] = v
    return ret_filtered

  def to_tuple(self):
    evaluated = self.get_evaluated()
    return tuple([evaluated[k] for k in self.get_state_names()])

  def __getitem__(self, item):
    return self.get_evaluated()[item]

  def compute_kernels(self, positions):
    # (x, y, dim)
    grid_node_coord = [[(i, j) for j in range(3)] for i in range(3)]
    grid_node_coord = np.array(grid_node_coord, dtype = np_precision)[None, :, :, :, None]
    frac = (positions - tf.floor(positions - 0.5))[:, None, None, :, :]
    assert frac.shape[3] == self.dim
    #print('frac', frac.shape)
    #print('grid_node_coord', grid_node_coord.shape)

    # (batch, x, y, dim, p) - (?, x, y, dim, ?)
    x = tf.abs(frac - grid_node_coord)
    #print('x', x.shape)

    mask = tf.cast(x < 0.5, tf_precision)
    y = mask * (0.75 - x * x) + (1 - mask) * (0.5 * (1.5 - x)**2)
    #print('y', y.shape)
    y = tf.reduce_prod(input_tensor=y, axis=3, keepdims=True)
    return y


class InitialSimulationState(SimulationState):
  """The subclass of :class:`SimulationState` that's used to represent the InitialState
  
  :param controller: a function mapping a class:`SimulationState` to a control actuation pressure matrix of batch_size x dim x dim x num_particles, applied at each state
  :type controller: function
  :param F_controller: a function mapping a class:`SimulationState` to a control force vector of batch_size x dim x num_particles, applied at each state
  :type F_controller: function
  """

  def __init__(self, sim, controller=None, F_controller = None):
    super().__init__(sim)
    dim = self.dim
    self.t = 0
    num_particles = sim.num_particles
    self.position = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, dim, num_particles], name='position')

    self.velocity = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, dim, num_particles], name='velocity')
    self.deformation_gradient = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, dim, dim, num_particles], name='dg')
    self.particle_mass = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='particle_mass')
    self.particle_volume = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='particle_volume')
    self.youngs_modulus = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='youngs_modulus')
    self.poissons_ratio = tf.compat.v1.placeholder(
        tf_precision, [self.sim.batch_size, 1, num_particles],
        name='poissons_ratio')
    self.grid_mass = tf.zeros(shape=self.grid_shape, dtype=tf_precision)
    self.grid_velocity = tf.zeros(shape=(*self.grid_shape[0:-1], dim), dtype=tf_precision) #TODO: this is a weird line
    if self.dim == 2:
      self.kernels = tf.zeros(shape=(self.sim.batch_size,
                                     kernel_size, kernel_size, 1, num_particles), dtype=tf_precision)
    self.step_count = tf.zeros(shape=(), dtype=np.int32)

    self.controller = controller
    self.F_controller = F_controller
    if controller is not None:
      self.actuation, self.debug = controller(self)
      self.actuation = matmatmul(self.deformation_gradient, matmatmul(self.actuation, transpose(self.deformation_gradient)))


class UpdatedSimulationState(SimulationState):
  """The subclass of :class:`SimulationState` that's used to represent the state after the first time step. This class also contains a CPU, pure TF implementation of the ChainQueen simulator.
  
  :param controller: a function mapping a class:`SimulationState` to a control actuation pressure matrix of batch_size x dim x dim x num_particles, applied at each state
  :type controller: function
  :param F_controller: a function mapping a class:`SimulationState` to a control force vector of batch_size x dim x num_particles, applied at each state
  :type F_controller: function
  """
  
  def cuda(self, sim, previous_state, controller, F_controller):
    """Calls the GPU simulator and handles substepping, and computes any other useful values to be returned in the memo.  Chains together steps.
>     
>     :param sim: The simulator object being used.
>     :type sim: :class:`Simulation`
>     :param previous_state: The state to be updated
>     :type previous_state: :class:`SimulationState`
>     :param controller: The pressure controller being used.
>     :type controller: function
>     :param F_controller: The force controller being used.
>     :type F_controller: function
>     """
  
  
    self.position = previous_state.position
    self.velocity = previous_state.velocity
    self.deformation_gradient = previous_state.deformation_gradient
    self.affine = previous_state.affine
    self.particle_mass = tf.identity(previous_state.particle_mass)
    self.particle_volume = tf.identity(previous_state.particle_volume)
    self.youngs_modulus = tf.identity(previous_state.youngs_modulus)
    self.acceleration = previous_state.acceleration
    self.poissons_ratio = tf.identity(previous_state.poissons_ratio)
    self.step_count = previous_state.step_count + 1
    
    self.grid_mass = tf.identity(previous_state.grid_mass)
    self.grid_velocity = tf.identity(previous_state.grid_velocity)

    #for i in range(self.dim):
    #  assert self.sim.gravity[i] == 0, "Non-zero gravity not supported"
    if controller:
      self.controller = controller
      self.actuation, self.debug = controller(self)
    else:
      self.controller = None
      self.actuation = np.zeros(shape=(self.sim.batch_size, self.dim, self.dim, self.sim.num_particles))

    self.t = previous_state.t + self.sim.dt

    num_cells = 1
    for i in range(self.dim):
      num_cells *= sim.grid_res[i]
    bc_normal = self.sim.bc_normal / (np.linalg.norm(self.sim.bc_normal, axis=self.dim + 1, keepdims=True) + 1e-10)
    bc = np.concatenate([bc_normal, self.sim.bc_parameter], axis=self.dim + 1).reshape(1, num_cells, self.dim + 1)
    bc = tf.constant(bc, dtype=tf_precision)
    
    if self.use_neohookean:
      neohookean_flag = 1
    else:
      neohookean_flag = 0

    self.position, self.velocity, self.deformation_gradient, self.affine, pressure, grid, grid_star = \
        mpm3d.mpm(position=self.position, velocity=self.velocity, 
                  deformation=self.deformation_gradient, affine=self.affine, dx=sim.dx,
                  dt=sim.dt/sim.substeps, gravity=sim.gravity, resolution=sim.grid_res,
                  youngsmodulus=self.youngs_modulus[:, 0], nu=self.poissons_ratio[:, 0], mass=self.particle_mass[:, 0],
                  V_p=sim.V_p, actuation=self.actuation, grid_bc=bc,
                  material_model=neohookean_flag, incompressibility_exp=self.incompressibility_exp)


    grid_reshaped = tf.reshape(grid, shape=(1, *sim.grid_res, self.dim + 1))
    if self.sim.dim == 2:
      self.grid_velocity = grid_reshaped[:, :, :, :-1]
      self.grid_mass = grid_reshaped[:, :, :, -1:]
    else:
      assert self.sim.dim == 3
      self.grid_velocity = grid_reshaped[:, :, :, :, :-1]
      self.grid_mass = grid_reshaped[:, :, :, :, -1:]


    if sim.damping != 0:
      self.velocity *= np.exp(-sim.damping * sim.dt)

    if F_controller:
      self.F = F_controller(self)
      self.velocity += self.F * sim.dt
      
    self.acceleration = (self.velocity - previous_state.velocity) * (1.0 / self.sim.dt)


  def __init__(self, sim, previous_state, controller=None, F_controller = None):
    """Contains an implementation for pure CPU, pure TF simulation"""
    super().__init__(sim)
    dim = self.dim
    if dim == 3 or self.use_cuda:
      # print("Running with cuda")
      self.cuda(sim, previous_state, controller=controller, F_controller = F_controller)
      return
    
    if F_controller is not None:
      raise Exception('F_controller is not defined with out cuda')


    self.controller = controller
    # 2D time integration
    self.particle_mass = tf.identity(previous_state.particle_mass)
    self.particle_volume = tf.identity(previous_state.particle_volume)
    self.youngs_modulus = tf.identity(previous_state.youngs_modulus)
    self.poissons_ratio = tf.identity(previous_state.poissons_ratio)
    self.step_count = previous_state.step_count + 1
    
    self.actuation = previous_state.actuation

    self.t = previous_state.t + self.sim.dt
    self.grid_velocity = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               dim), dtype = tf_precision)
    
    position = previous_state.position
    
    minimum_positions = np.zeros(shape=previous_state.position.shape, dtype=np_precision)
    minimum_positions[:, :, :] = self.sim.dx * 2
    maximum_positions = np.zeros(shape=previous_state.position.shape, dtype=np_precision)
    for i in range(dim):
      maximum_positions[:, i, :] = (self.sim.grid_res[i] - 2) * self.sim.dx
    # Safe guard
    position = tf.clip_by_value(position, minimum_positions, maximum_positions)

    # Rasterize mass and velocity
    base_indices = tf.cast(
        tf.floor(position * sim.inv_dx - 0.5), tf.int32)
    base_indices = tf.transpose(a=base_indices, perm=[0, 2, 1])
    batch_size = self.sim.batch_size
    num_particles = sim.num_particles

    # Add the batch size indices
    base_indices = tf.concat(
        [
            tf.zeros(shape=(batch_size, num_particles, 1), dtype=tf.int32),
            base_indices,
        ],
        axis=2)

    # print('base indices', base_indices.shape)
    self.grid_mass = tf.zeros(dtype = tf_precision, shape=(batch_size, self.sim.grid_res[0],
                                     self.sim.grid_res[1], 1))

    # Compute stress tensor (Kirchhoff stress instead of First Piola-Kirchhoff stress)
    self.deformation_gradient = previous_state.deformation_gradient

    # Lame parameters
    mu = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
    lam = self.youngs_modulus * self.poissons_ratio / ((
        1 + self.poissons_ratio) * (1 - 2 * self.poissons_ratio))
    # (b, 1, p) -> (b, 1, 1, p)
    mu = mu[:, :, None, :]
    lam = lam[:, :, None, :]
    # Corotated elasticity
    # P(F) = dPhi/dF(F) = 2 mu (F-R) + lambda (J-1)JF^-T

    r, s = polar_decomposition(self.deformation_gradient)
    j = determinant(self.deformation_gradient)[:, None, None, :]

    # Note: stress_tensor here is right-multiplied by F^T
    self.stress_tensor1 = 2 * mu * matmatmul(
        self.deformation_gradient - r, transpose(self.deformation_gradient))

    if previous_state.controller:
      self.stress_tensor1 += previous_state.actuation

    self.stress_tensor2 = lam * (j - 1) * j * identity_matrix

    self.stress_tensor = self.stress_tensor1 + self.stress_tensor2

    # Rasterize momentum and velocity
    # ... and apply gravity

    self.grid_velocity = tf.zeros(shape=(batch_size, self.sim.grid_res[0],
                                         self.sim.grid_res[1], dim), dtype = tf_precision)

    self.kernels = self.compute_kernels(position * sim.inv_dx)
    assert self.kernels.shape == (batch_size, kernel_size, kernel_size, 1, num_particles)

    self.velocity = previous_state.velocity

    # Quadratic B-spline kernel
    for i in range(kernel_size):
      for j in range(kernel_size):
        delta_indices = np.zeros(
            shape=(self.sim.batch_size, 1, dim + 1), dtype=np.int32)

        for b in range(batch_size):
          delta_indices[b, 0, :] = [b, i, j]
        self.grid_mass = self.grid_mass + tf.scatter_nd(
            shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], 1),
            indices=base_indices + delta_indices,
            updates=tf.transpose(a=(self.particle_mass * self.kernels[:, i, j, :, :]), perm=[0, 2, 1]))

        # (b, dim, p)
        delta_node_position = np.array([i, j], dtype = np_precision)[None, :, None]
        # xi - xp
        offset = (tf.floor(position * sim.inv_dx - 0.5) +
                  tf.cast(delta_node_position, tf_precision) -
                  position * sim.inv_dx) * sim.dx

        grid_velocity_contributions = self.particle_mass * self.kernels[:, i, j, :] * (
            self.velocity + matvecmul(previous_state.affine, offset))
        grid_force_contributions = self.particle_volume * self.kernels[:, i, j, :] * (
            matvecmul(self.stress_tensor, offset) *
            (-4 * self.sim.dt * self.sim.inv_dx * self.sim.inv_dx))
        self.grid_velocity = self.grid_velocity + tf.scatter_nd(
            shape=(batch_size, self.sim.grid_res[0], self.sim.grid_res[1], dim),
            indices=base_indices + delta_indices,
            updates=tf.transpose(a=grid_velocity_contributions + grid_force_contributions, perm=[0, 2, 1]))
    assert self.grid_mass.shape == (batch_size, self.sim.grid_res[0],
                                    self.sim.grid_res[1],
                                    1), 'shape={}'.format(self.grid_mass.shape)

    self.grid_velocity += self.grid_mass * np.array(
        self.sim.gravity, dtype = np_precision)[None, None, None, :] * self.sim.dt
    self.grid_velocity = self.grid_velocity / tf.maximum(np_precision(1e-30), self.grid_mass)

    sticky_mask = tf.cast(self.sim.bc_parameter == -1, tf_precision)
    self.grid_velocity *= (1 - sticky_mask)
    
    mask = tf.cast(
        tf.reduce_sum(input_tensor=self.sim.bc_normal**2, axis=3, keepdims=True) != 0,
        tf_precision)
    normal_component_length = tf.reduce_sum(
        input_tensor=self.grid_velocity * self.sim.bc_normal, axis=3, keepdims=True)
    perpendicular_component = self.grid_velocity - self.sim.bc_normal * normal_component_length
    perpendicular_component_length = tf.sqrt(
        tf.reduce_sum(input_tensor=perpendicular_component**2, axis=3, keepdims=True) + 1e-7)
    normalized_perpendicular_component = perpendicular_component / perpendicular_component_length
    perpendicular_component_length = tf.maximum(perpendicular_component_length +
                                                tf.minimum(normal_component_length, 0) * self.sim.bc_parameter, 0)
    projected_velocity = sim.bc_normal * tf.maximum(
        normal_component_length,
        0) + perpendicular_component_length * normalized_perpendicular_component
    self.grid_velocity = self.grid_velocity * (
        1 - mask) + mask * projected_velocity

    # Resample velocity and local affine velocity field
    self.velocity *= 0
    for i in range(kernel_size):
      for j in range(kernel_size):
        delta_indices = np.zeros(
          shape=(self.sim.batch_size, 1, dim + 1), dtype=np.int32)
        for b in range(batch_size):
          delta_indices[b, 0, :] = [b, i, j]


        #print('indices', (base_indices + delta_indices).shape)
        grid_v = tf.transpose(a=tf.gather_nd(
            params=self.grid_velocity,
            indices=base_indices + delta_indices), perm=[0, 2, 1])
        self.velocity = self.velocity + grid_v * self.kernels[:, i, j, :]

        delta_node_position = np.array([i, j], dtype = np_precision)[None, :, None]

        # xi - xp
        offset = (tf.floor(position * sim.inv_dx - 0.5) +
                  tf.cast(delta_node_position, tf_precision) -
                  position * sim.inv_dx) * sim.dx
        assert offset.shape == position.shape
        weighted_node_velocity = grid_v * self.kernels[:, i, j, :]
        # weighted_node_velocity = tf.transpose(weighted_node_velocity, perm=[0, 2, 1])
        self.affine += outer_product(weighted_node_velocity, offset)

    self.affine *= 4 * sim.inv_dx * sim.inv_dx
    dg_change = identity_matrix + self.sim.dt * self.affine
    if not use_apic:
      self.affine *= 0
    #print(dg_change.shape)
    #print(previous_state.deformation_gradient)
    self.deformation_gradient = matmatmul(dg_change,
                                          previous_state.deformation_gradient)

    # Advection
    self.position = position + self.velocity * self.sim.dt
    assert self.position.shape == previous_state.position.shape
    assert self.velocity.shape == previous_state.velocity.shape
    self.acceleration = (self.velocity - previous_state.velocity) * (1 / self.sim.dt)
    
    if sim.damping != 0:
      self.velocity *= np.exp(-sim.damping * sim.dt)

