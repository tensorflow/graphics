# Copyright 2021 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the Spatial Transformer Layer, for 3D quaternions.

@misc{jaderberg2016spatial,
      title={Spatial Transformer Networks},
      author={Max Jaderberg et al.},
      year={2016},
      eprint={1506.02025},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

NOTE: Nearest neighbor interpolation is not available during training,
because of non-derivable operations.
"""

import tensorflow as tf
from tensorflow_graphics.util import export_api

def repeat(x, num_reps):
  """ Repeat input multiple times """
  num_reps = tf.cast(num_reps, dtype=tf.int32)
  if tf.rank(x) == 1:
    x = tf.expand_dims(x, axis=1)
  return tf.tile(x, multiples=(1, num_reps))

def from_quaternion(quaternion):
  """ Return the 3D rotation matrix from quaternion [a, b, c, w]. """
  x, y, z, w = tf.unstack(quaternion, axis=-1)
  tx = 2.0 * x
  ty = 2.0 * y
  tz = 2.0 * z
  twx = tx * w
  twy = ty * w
  twz = tz * w
  txx = tx * x
  txy = ty * x
  txz = tz * x
  tyy = ty * y
  tyz = tz * y
  tzz = tz * z
  r = tf.stack((1.0 - (tyy + tzz), txy - twz, txz + twy,
                txy + twz, 1.0 - (txx + tzz), tyz - twx,
                txz - twy, tyz + twx, 1.0 - (txx + tyy)),
               axis=-1)

  return tf.reshape(r, shape=(-1, 3, 3))

def matrix_from_params(transfos):
  """ Get the augmented transformation matrix from affine parameters. """
  num_batch = tf.shape(transfos)[0]

  # scaling [q0, q1, q2, q3, tx, ty, tz, sx, sy, sz]
  if tf.shape(transfos)[-1] == 10:
    scaling = tf.linalg.diag(transfos[:, -3:])
  else:
    scaling = tf.eye(num_rows=3, batch_shape=(num_batch,))
  # rotation [q0, q1, q2, q3]
  rotation = from_quaternion(transfos[..., :4])
  # theta = S @ R
  scale_rotation = tf.linalg.matmul(scaling, rotation)
  # translation [q0, q1, q2, q3, tx, ty, tz]
  if tf.shape(transfos)[-1] == 7:
    thetas = tf.concat([scale_rotation, transfos[:, 4:7, tf.newaxis]], axis=2)
  else:
    thetas = tf.concat([scale_rotation, tf.zeros((num_batch, 3, 1))], axis=2)

  return thetas

class SpatialTransformer3D(tf.keras.layers.Layer):
  """ The 3D Spatial Transformer derivable layer."""
  def __init__(self
               , min_ref_grid=[-1., -1., -1.]
               , max_ref_grid=[1., 1., 1.]
               , interp_method="bilinear"
               , padding_mode="min"
               , **kwargs):
    """Constructs a 3D Spatial Transformer layer.

    Args:
      min_ref_grid: `list` of `float`.
        The starting points to define the resampling grid
        for each spatial dimension (default: [-1., -1., -1.]).
      max_ref_grid: `list` of `float`.
        The end points to define the resampling grid
        for each spatial dimension (default: [1., 1., 1.]).
      interp_method: `string` from `"bilinear"` or `"nn"`.
        `"bilinear"` takes the weighted sum of each neighboring pixel,
        `"nn"` takes instead the nearest pixel (default: `"bilinear"`).
      padding_mode: `string` between `"border"`, `"zeros"` or `"min"`.
        It defines which default value should be used for pixels that are
        outside the grid after the transformation.
        `"border"` to use the same value as the border,
        `"zeros"` to nullify them,
        `"min"` to use the minimum value from the input tensor (default).
      **kwargs: Additional keyword arguments passed to the base layer.
    """
    super(SpatialTransformer3D, self).__init__(**kwargs)
    self.min_ref_grid = tf.constant(min_ref_grid, dtype=tf.float32)
    self.max_ref_grid = tf.constant(max_ref_grid, dtype=tf.float32)
    self.interp_method = tf.constant(interp_method, dtype=tf.string)
    self.padding_mode = tf.constant(padding_mode, dtype=tf.string)

  def build(self, input_shape):
    num_dims = input_shape[0].ndims - 2
    shape_grid = tf.shape(self.min_ref_grid)[0]

    # inputs is a list of size 2
    tf.debugging.assert_equal(len(input_shape), 2)
    # transformation size is at least 4 (quaternion),
    # but no more than 10 (quaternion + translation + scale)
    tf.debugging.assert_greater_equal(input_shape[1][-1], 4)
    tf.debugging.assert_less_equal(input_shape[1][-1], 10)
    # interpolation method
    valid_interpolation = tf.constant(["bilinear", "nn"], dtype=tf.string)
    check_interpolation = tf.math.equal(self.interp_method, valid_interpolation)
    check_interpolation = tf.reduce_any(check_interpolation)
    tf.debugging.assert_equal(check_interpolation
                              , True
                              , message="{} must be anything between {}, but "
                                        "is {}".format("interp_method"
                                                       , valid_interpolation
                                                       , self.interp_method))
    # padding method
    valid_padding = tf.constant(["border", "zeros", "min"], dtype=tf.string)
    check_padding = tf.math.equal(self.padding_mode, valid_padding)
    check_padding = tf.reduce_any(check_padding)
    tf.debugging.assert_equal(check_padding
                              , True
                              , message="{} must be anything between {}, "
                                        "but is {}".format("padding_mode"
                                                           , valid_padding
                                                           , self.padding_mode))

    # validate reference grid values
    if tf.math.not_equal(num_dims, shape_grid):
      if tf.math.equal(shape_grid, 1):
        self.min_ref_grid = self.min_ref_grid[0] \
                            * tf.ones(num_dims, dtype=tf.float32)
        self.max_ref_grid = self.max_ref_grid[0] \
                            * tf.ones(num_dims, dtype=tf.float32)
      else:
        self.min_ref_grid = (-1) * tf.ones(num_dims, dtype=tf.float32)
        self.max_ref_grid = tf.ones(num_dims, dtype=tf.float32)
    # cannot train if method is nearest neighbor
    if tf.math.logical_and(self.trainable == tf.constant(True)
                           , self.interp_method == tf.constant("nn")):
      raise Exception("Cannot train with nearest-neighbor interpolator "
                      "because it is not derivable!")
    super(SpatialTransformer3D, self).build(input_shape)

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def get_config(self):
    return {
        'min_ref_grid': self.min_ref_grid,
        'max_ref_grid': self.max_ref_grid,
        'interp_method': self.interp_method,
        'padding_mode': self.padding_mode
    }

  def call(self, inputs, **kwargs):
    """Call a 3D Spatial Transformer layer.

    Args:
      inputs: `list` of two Tensor with shape `[A0, W, H, D, C]` and `[A0, N]`.
        The first element of this list is the input 3D volume to be resampled,
        the second element is the normalized quaternion to apply with optionnal
        translations and scaling `[a, b, c, w, tx, ty, tz, sx, sy, sz]`, where
        the quaternion Q = a*i + b*j + c*k + w.
      training: flag to control batch normalization update statistics.

    Returns:
      Tensor with shape `[A0, W, H, D, C]`.
    """
    img, transfos = inputs
    output = self._resample(img, transfos)

    return output

  def _resample(self, img, transfos):
    input_shape = tf.shape(img)
    ref_size = input_shape[1:-1]
    ref_size_xyz = tf.concat([ref_size[1::-1], ref_size[2:]], axis=0)

    input_transformed = self._transform_grid(ref_size_xyz
                                             , transfos=transfos
                                             , min_ref_grid=self.min_ref_grid
                                             , max_ref_grid=self.max_ref_grid)
    input_transformed = self._interpolate(im=img
                                          , points=input_transformed
                                          , min_ref_grid=self.min_ref_grid
                                          , max_ref_grid=self.max_ref_grid
                                          , method=self.interp_method
                                          , padding_mode=self.padding_mode)
    output = tf.reshape(input_transformed, shape=input_shape)

    return output

  def _transform_grid(self, ref_size_xyz, transfos, min_ref_grid, max_ref_grid):
    num_batch = tf.shape(transfos)[0]
    num_elems = tf.reduce_prod(ref_size_xyz)
    thetas = matrix_from_params(transfos)

    # grid creation from volume affine
    mz, my, mx = tf.meshgrid(tf.linspace(min_ref_grid[2]
                                         , max_ref_grid[2]
                                         , ref_size_xyz[2])
                             , tf.linspace(min_ref_grid[1]
                                           , max_ref_grid[1]
                                           , ref_size_xyz[1])
                             , tf.linspace(min_ref_grid[0]
                                           , max_ref_grid[0]
                                           , ref_size_xyz[0])
                             , indexing='ij')

    # preparing grid for quaternion rotation
    grid = tf.concat([tf.reshape(mx, (1, -1))
                      , tf.reshape(my, (1, -1))
                      , tf.reshape(mz, (1, -1))], axis=0)
    grid = tf.expand_dims(grid, axis=0)
    grid = tf.tile(grid, (num_batch, 1, 1))

    # preparing grid for augmented transformation
    grid = tf.concat([grid, tf.ones((num_batch, 1, num_elems))], axis=1)

    return tf.linalg.matmul(thetas, grid)

  def _interpolate(self
                   , im, points
                   , min_ref_grid
                   , max_ref_grid
                   , method="bilinear"
                   , padding_mode="zeros"):
    num_batch = tf.shape(im)[0]
    vol_shape_xyz = tf.concat([tf.shape(im)[1:-1][1::-1]
                               , tf.shape(im)[1:-1][2:]], axis=0)
    vol_shape_xyz = tf.cast(vol_shape_xyz, dtype=tf.float32)
    width = vol_shape_xyz[0]
    height = vol_shape_xyz[1]
    depth = vol_shape_xyz[2]
    width_i = tf.cast(width, dtype=tf.int32)
    height_i = tf.cast(height, dtype=tf.int32)
    depth_i = tf.cast(depth, dtype=tf.int32)
    channels = tf.shape(im)[-1]
    num_row_major = tf.cast(tf.math.cumprod(vol_shape_xyz), dtype=tf.int32)
    shape_output = tf.stack([num_batch, num_row_major[-1], 1])
    zero = tf.zeros([], dtype=tf.float32)
    zero_i = tf.zeros([], dtype=tf.int32)
    ibatch = repeat(num_row_major[-1] * tf.range(num_batch, dtype=tf.int32)
                    , num_row_major[-1])
    output = tf.zeros(shape_output, dtype=tf.float32)
    valid = tf.ones([])

    # scale positions to [0, width/height - 1]
    coeff_x = (width - 1.)/(max_ref_grid[0] - min_ref_grid[0])
    coeff_y = (height - 1.)/(max_ref_grid[1] - min_ref_grid[1])
    coeff_z = (depth - 1.)/(max_ref_grid[2] - min_ref_grid[2])
    ix = (coeff_x * points[:, 0, :]) - (coeff_x *  min_ref_grid[0])
    iy = (coeff_y * points[:, 1, :]) - (coeff_y *  min_ref_grid[1])
    iz = (coeff_z * points[:, 2, :]) - (coeff_z *  min_ref_grid[2])

    # zeros and min padding mode, for positions outside of refrence grid
    if tf.math.logical_or(tf.math.equal(padding_mode, "zeros")
                          , tf.math.equal(padding_mode, "min")):
      valid = tf.less_equal(ix, width - 1.) & tf.greater_equal(ix, zero) \
              & tf.less_equal(iy, height - 1.) & tf.greater_equal(iy, zero) \
              & tf.less_equal(iz, depth - 1.) & tf.greater_equal(iz, zero)
      valid = tf.expand_dims(tf.cast(valid, dtype=tf.float32), -1)

    # for bilinear interpolation, calculate each area between corners
    # and positions to get each pixel's weight
    if tf.math.equal(method, tf.constant("bilinear", dtype=tf.string)):
      # get north-west-top corner indexes based on the scaled positions
      ix_nwt = tf.clip_by_value(tf.floor(ix), zero, width - 1.)
      iy_nwt = tf.clip_by_value(tf.floor(iy), zero, height - 1.)
      iz_nwt = tf.clip_by_value(tf.floor(iz), zero, depth - 1.)
      ix_nwt_i = tf.cast(ix_nwt, dtype=tf.int32)
      iy_nwt_i = tf.cast(iy_nwt, dtype=tf.int32)
      iz_nwt_i = tf.cast(iz_nwt, dtype=tf.int32)

      #gettings all offsets to create corners
      offset_corner = tf.constant([[0., 0., 0.]
                                   , [0., 0., 1.]
                                   , [0., 1., 0.]
                                   , [0., 1., 1.]
                                   , [1., 0., 0.]
                                   , [1., 0., 1.]
                                   , [1., 1., 0.]
                                   , [1., 1., 1.]], dtype=tf.float32)
      offset_corner_i = tf.cast(offset_corner, dtype=tf.int32)

      for c in range(8):
        # getting all corner indexes from north-west-top corner
        ix_c = ix_nwt + offset_corner[-c - 1, 0]
        iy_c = iy_nwt + offset_corner[-c - 1, 1]
        iz_c = iz_nwt + offset_corner[-c - 1, 2]
        # area is computed using the opposite corner
        nc = tf.expand_dims(tf.abs((ix - ix_c) * (iy - iy_c) * (iz - iz_c)), -1)
        # current corner position
        ix_c = ix_nwt_i + offset_corner_i[c, 0]
        iy_c = iy_nwt_i + offset_corner_i[c, 1]
        iz_c = iz_nwt_i + offset_corner_i[c, 2]
        # gather input image values from corners idx,
        # and calculate weighted pixel value
        offset_xy = num_row_major[0] \
                    * tf.clip_by_value(iy_c, zero_i, height_i - 1)
        offset_xyz = num_row_major[1] \
                    * tf.clip_by_value(iz_c, zero_i, depth_i - 1)
        idx_c = ibatch + tf.clip_by_value(ix_c, zero_i, width_i - 1) \
                + offset_xy + offset_xyz
        ic = tf.gather(tf.reshape(im, [-1, channels]), idx_c)

        output += nc * ic
    # otherwise for nearest neighbor, just get the nearest corner
    elif tf.math.equal(method, tf.constant("nn", dtype=tf.string)):
      # get rounded indice corner based on the scaled positions
      ix_nn = tf.cast(tf.clip_by_value(tf.round(ix), zero, width - 1.)
                      , dtype=tf.int32)
      iy_nn = tf.cast(tf.clip_by_value(tf.round(iy), zero, height - 1.)
                      , dtype=tf.int32)
      iz_nn = tf.cast(tf.clip_by_value(tf.round(iz), zero, depth - 1.)
                      , dtype=tf.int32)

      # gather input pixel values from nn corner indexes
      idx_nn = ibatch + ix_nn
      idx_nn = idx_nn + num_row_major[0] * iy_nn + num_row_major[1] * iz_nn
      output = tf.gather(tf.reshape(im, [-1, channels]), idx_nn)

    # padding mode
    if tf.math.equal(padding_mode, tf.constant("zeros", dtype=tf.string)):
      output = output * valid
    elif tf.math.equal(padding_mode, tf.constant("min", dtype=tf.string)):
      output = output * valid + tf.reduce_min(im) * (1. - valid)
    elif tf.math.equal(padding_mode, tf.constant("border", dtype=tf.string)):
      output = output

    return output

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
