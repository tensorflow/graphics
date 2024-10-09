import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import os
import numpy as np

file_dir = os.path.dirname(os.path.realpath(__file__))
MPM_module = tf.load_op_library(os.path.join(file_dir, '../../build/libtaichi_tf_differentiable_mpm.so'))

mpm = MPM_module.mpm

def normalize_grid(grid, res, gravity, dt):
  np_res = np.array(res)
  dim = np_res.shape[0]
  assert len(grid.shape) == 3
  assert dim in [2, 3], 'Dimension must be 2 or 3!'
  batch_size, num_cells, dim_1 = grid.shape
  assert dim_1 == dim + 1, dim_1
  assert num_cells == np_res.prod()

  grid_v = grid[:, :, :dim]
  grid_m = grid[:, :, dim:]
  grid_v += grid_m * np.array(gravity)[None, None, :] * dt
  grid_v = grid_v / tf.maximum(1e-30, grid_m)

  '''

  sticky_mask = tf.cast(bc_parameter == -1, tf.float32)
  grid_v *= (1 - sticky_mask)

  mask = tf.cast(
        tf.reduce_sum(bc_normal**2, axis=3, keepdims=True) != 0,
        tf.float32)
  normal_component_length = tf.reduce_sum(
      grid_v * bc_normal, axis=3, keepdims=True)
  perpendicular_component = grid_v - bc_normal * normal_component_length
  perpendicular_component_length = tf.sqrt(
      tf.reduce_sum(perpendicular_component**2, axis=3, keepdims=True) + 1e-7)
  normalized_perpendicular_component = perpendicular_component / tf.maximum(
      perpendicular_component_length, 1e-7)
  perpendicular_component_length = tf.sign(perpendicular_component_length) * \
                                   tf.maximum(tf.abs(perpendicular_component_length) +
                                              tf.minimum(normal_component_length, 0) * bc_parameter, 0)
  projected_velocity = bc_normal * tf.maximum(
      normal_component_length,
      0) + perpendicular_component_length * normalized_perpendicular_component
  grid_v = grid_v * (
      1 - mask) + mask * projected_velocity
  '''

  grid = tf.concat([grid_v, grid_m], axis = 2)

  return grid

@ops.RegisterGradient("Mpm")
def _mpm_grad_cc(op, *grads):
  """The gradient of the MPM function, as defined by the kernel described in the src directory, for :func:`mpm`"""
  attr_labels = ['dt', 'dx', 'gravity', 'resolution', 'V_p', 'material_model', 'incompressibility_exp']
  attrs = {}
  for st in attr_labels:
    attrs[st] = op.get_attr(st)
    

  tensors = list(op.inputs) + list(op.outputs) + list(grads)
  return MPM_module.mpm_grad(*tensors, **attrs)
