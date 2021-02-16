# Copyright 2020 The TensorFlow Authors
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
"""This module implements image matting functionalities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_graphics.math import vector as tfg_vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _shape(batch_shape, *shapes):
  """Creates a new shape concatenating batch_shape and shapes.

  Args:
    batch_shape: A Tensor or list/tuple containing the batch dimensions.
    *shapes: A tuple containing new dimensions to append to the final shape.

  Returns:
    A Tensor containing the final shape dimensions.
  """
  return tf.concat((batch_shape, shapes), axis=-1)


def _quadratic_form(matrix, vector):
  """Computes the quadratic form between a matrix and a vector.

  The quadratic form between a matrix A and a vector x can be written as
  Q(x) = <x,Ax>, where <.,.> is the dot product operator.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, C, C]`.
    vector: A tensor of shape `[A1, ..., An, 1, C]`.

  Returns:
    A tensor of shape `[A1, ..., An, 1]`.
  """
  vector_matrix = tf.matmul(vector, matrix)
  vector_matrix_vector = tf.matmul(vector_matrix, vector, transpose_b=True)
  return vector_matrix_vector


def _image_patches(image, size):
  """Extracts square image patches.

  Args:
    image: A tensor of shape `[B, H, W, C]`.
    size: The size of the square patches.

  Returns:
    A tensor of shape `[B, H - pad, W - pad, C * size^2]`, where `pad` is
    `size - 1`.
  """
  return tf.image.extract_patches(
      image,
      sizes=(1, size, size, 1),
      strides=(1, 1, 1, 1),
      rates=(1, 1, 1, 1),
      padding="VALID")


def _image_average(image, size):
  """Computes average over image patches.

  Args:
    image: A tensor of shape `[B, H, W, C]`.
    size: The size of the square patches.

  Returns:
    A tensor of shape `[B, H - pad, W - pad, C]`, where `pad` is `size - 1`.
  """
  return tf.nn.avg_pool2d(
      input=image,
      ksize=(1, size, size, 1),
      strides=(1, 1, 1, 1),
      padding="VALID")


def build_matrices(image, size=3, eps=1e-5, name="matting_build_matrices"):
  """Generates the closed form matting Laplacian.

  Generates the closed form matting Laplacian as proposed by Levin et
  al. in "A Closed Form Solution to Natural Image Matting". This function also
  return the pseudo-inverse matrix allowing to retrieve the matting linear
  coefficient.

  Args:
    image: A tensor of shape `[B, H, W, C]`.
    size: An `int` representing the size of the patches used to enforce
      smoothness.
    eps: A small number of type `float` to regularize the problem.
    name: A name for this op. Defaults to "matting_build_matrices".

  Returns:
    A tensor of shape `[B, H - pad, W - pad, size^2, size^2]` containing
    the matting Laplacian matrices. A tensor of shape
    `[B, H - pad, W - pad, C + 1, size^2]` containing the pseudo-inverse
    matrices which can be used to retrieve the matting linear coefficients.
    The padding `pad` is equal to `size - 1`.

  Raises:
    ValueError: If `image` is not of rank 4.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)
    eps = tf.constant(value=eps, dtype=image.dtype)

    shape.check_static(image, has_rank=4)
    if size % 2 == 0:
      raise ValueError("The patch size is expected to be an odd value.")

    pixels = size**2
    channels = tf.shape(input=image)[-1]
    dtype = image.dtype
    # Extracts image patches.
    patches = _image_patches(image, size)
    batches = tf.shape(input=patches)[:-1]
    patches = tf.reshape(patches, shape=_shape(batches, pixels, channels))
    # Creates the data matrix block.
    ones = tf.ones(shape=_shape(batches, pixels, 1), dtype=dtype)
    affine = tf.concat((patches, ones), axis=-1)
    # Creates the regularizer matrix block.
    diag = tf.sqrt(eps) * tf.eye(channels, batch_shape=(1, 1, 1), dtype=dtype)
    zeros = tf.zeros(shape=_shape((1, 1, 1), channels, 1), dtype=dtype)
    regularizer = tf.concat((diag, zeros), axis=-1)
    regularizer = tf.tile(regularizer, multiples=_shape(batches, 1, 1))
    # Creates a matrix concatenating the data and regularizer blocks.
    mat = tf.concat((affine, regularizer), axis=-2)
    # Builds the pseudo inverse and the laplacian matrices.
    inverse = tf.linalg.inv(tf.matmul(mat, mat, transpose_a=True))
    inverse = asserts.assert_no_infs_or_nans(inverse)
    pseudo_inverse = tf.matmul(inverse, affine, transpose_b=True)
    identity = tf.eye(num_rows=pixels, dtype=dtype)
    laplacian = identity - tf.matmul(affine, pseudo_inverse)
    return laplacian, pseudo_inverse


def linear_coefficients(matte,
                        pseudo_inverse,
                        name="matting_linear_coefficients"):
  """Computes the matting linear coefficients.

  Computes the matting linear coefficients (a, b) based on the `pseudo_inverse`
  generated by the `build_matrices` function which implements the approach
  proposed by Levin et al. in "A Closed Form Solution to Natural Image Matting".

  Args:
    matte: A tensor of shape `[B, H, W, 1]`.
    pseudo_inverse: A tensor of shape `[B, H - pad, W - pad, C + 1, size^2]`
      containing the pseudo-inverse matrices computed by the `build_matrices`
      function, where `pad` is equal to `size - 1` and `size` is the patch size
      used to compute this tensor.
    name: A name for this op. Defaults to "matting_linear_coefficients".

  Returns:
    A tuple contraining two Tensors for the linear coefficients (a, b) of shape
    `[B, H, W, C]` and `[B, H, W, 1]`.

  Raises:
    ValueError: If the last dimension of `matte` is not 1. If `matte` is not
    of rank 4. If `pseudo_inverse` is not of rank 5. If `B` is different
    between `matte` and `pseudo_inverse`.
  """
  with tf.name_scope(name):
    matte = tf.convert_to_tensor(value=matte)
    pseudo_inverse = tf.convert_to_tensor(value=pseudo_inverse)

    pixels = tf.compat.dimension_value(pseudo_inverse.shape[-1])
    shape.check_static(matte, has_rank=4, has_dim_equals=(-1, 1))
    shape.check_static(pseudo_inverse, has_rank=5)
    shape.compare_batch_dimensions(
        tensors=(matte, pseudo_inverse),
        last_axes=0,
        broadcast_compatible=False)

    size = np.sqrt(pixels)
    # Computes the linear coefficients.
    patches = tf.expand_dims(_image_patches(matte, size), axis=-1)
    coeffs = tf.squeeze(tf.matmul(pseudo_inverse, patches), axis=-1)
    # Averages the linear coefficients over patches.
    height = tf.shape(input=coeffs)[1]
    width = tf.shape(input=coeffs)[2]
    ones = tf.ones(shape=_shape((1,), height, width, 1), dtype=matte.dtype)
    height = tf.shape(input=matte)[1] + size - 1
    width = tf.shape(input=matte)[2] + size - 1
    coeffs = tf.image.resize_with_crop_or_pad(coeffs, height, width)
    ones = tf.image.resize_with_crop_or_pad(ones, height, width)
    coeffs = _image_average(coeffs, size) / _image_average(ones, size)
    return tf.split(coeffs, (-1, 1), axis=-1)


def loss(matte, laplacian, name="matting_loss"):
  """Computes the matting loss function based on the matting Laplacian.

  Computes the matting loss function based on the `laplacian` generated by the
  `build_matrices` function which implements the approach proposed by Levin
  et al. in "A Closed Form Solution to Natural Image Matting".

  Args:
    matte: A tensor of shape `[B, H, W, 1]`.
    laplacian: A tensor of shape `[B, H - pad, W - pad, size^2, size^2]`
      containing the Laplacian matrices computed by the `build_matrices`
      function, where `pad` is equal to `size - 1` and `size` is the patch size
      used to compute this tensor.
    name: A name for this op. Defaults to "matting_loss".

  Returns:
    A tensor containing a scalar value defining the matting loss.

  Raises:
    ValueError: If the last dimension of `matte` is not 1. If `matte` is not
    of rank 4. If the last two dimensions of `laplacian` are not of the
    same size. If `laplacian` is not of rank 5. If `B` is different
    between `matte` and `laplacian`.
  """
  with tf.name_scope(name):
    matte = tf.convert_to_tensor(value=matte)
    laplacian = tf.convert_to_tensor(value=laplacian)

    pixels = tf.compat.dimension_value(laplacian.shape[-1])
    shape.check_static(matte, has_rank=4, has_dim_equals=(-1, 1))
    shape.check_static(laplacian, has_rank=5, has_dim_equals=(-2, pixels))
    shape.compare_batch_dimensions(
        tensors=(matte, laplacian), last_axes=0, broadcast_compatible=False)

    size = np.sqrt(pixels)
    patches = tf.expand_dims(_image_patches(matte, size), axis=-2)
    losses = _quadratic_form(laplacian, patches)
    return tf.reduce_mean(input_tensor=losses)


def reconstruct(image, coeff_mul, coeff_add, name="matting_reconstruct"):
  """Reconstruct the matte from the image using the linear coefficients.

  Reconstruct the matte from the image using the linear coefficients (a, b)
  returned by the linear_coefficients function.

  Args:
    image: A tensor of shape `[B, H, W, C]` .
    coeff_mul: A tensor of shape `[B, H, W, C]` representing the multiplicative
      part of the linear coefficients.
    coeff_add: A tensor of shape `[B, H, W, 1]` representing the additive part
      of the linear coefficients.
    name: A name for this op. Defaults to "matting_reconstruct".

  Returns:
    A tensor of shape `[B, H, W, 1]` containing the mattes.

  Raises:
    ValueError: If `image`, `coeff_mul`, or `coeff_add` are not of rank 4. If
    the last dimension of `coeff_add` is not 1. If the batch dimensions of
    `image`, `coeff_mul`, and `coeff_add` do not match.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)
    coeff_mul = tf.convert_to_tensor(value=coeff_mul)
    coeff_add = tf.convert_to_tensor(value=coeff_add)

    shape.check_static(image, has_rank=4)
    shape.check_static(coeff_mul, has_rank=4)
    shape.check_static(coeff_add, has_rank=4, has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(image, coeff_mul), last_axes=-1, broadcast_compatible=False)
    shape.compare_batch_dimensions(
        tensors=(image, coeff_add), last_axes=-2, broadcast_compatible=False)

    return tfg_vector.dot(coeff_mul, image) + coeff_add


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
