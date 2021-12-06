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
"""This module implements routines required for spherical harmonics lighting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.math import math_helpers
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util.type_alias import TensorLike


def integration_product(
    harmonics1: TensorLike,
    harmonics2: TensorLike,
    keepdims: bool = True,
    name: str = "spherical_harmonics_convolution") -> TensorLike:
  """Computes the integral of harmonics1.harmonics2 over the sphere.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    harmonics1: A tensor of shape `[A1, ..., An, C]`, where the last dimension
      represents spherical harmonics coefficients.
    harmonics2: A tensor of shape `[A1, ..., An, C]`, where the last dimension
      represents spherical harmonics coefficients.
    keepdims: If True, retains reduced dimensions with length 1.
    name: A name for this op. Defaults to "spherical_harmonics_convolution".

  Returns:
    A tensor of shape `[A1, ..., An]` containing scalar values resulting from
    integrating the product of the spherical harmonics `harmonics1` and
    `harmonics2`.

  Raises:
    ValueError: if the last dimension of `harmonics1` is different from the last
    dimension of `harmonics2`.
  """
  with tf.name_scope(name):
    harmonics1 = tf.convert_to_tensor(value=harmonics1)
    harmonics2 = tf.convert_to_tensor(value=harmonics2)

    shape.compare_dimensions(
        tensors=(harmonics1, harmonics2),
        axes=-1,
        tensor_names=("harmonics1", "harmonics2"))
    shape.compare_batch_dimensions(
        tensors=(harmonics1, harmonics2),
        last_axes=-2,
        tensor_names=("harmonics1", "harmonics2"),
        broadcast_compatible=True)

    return vector.dot(harmonics1, harmonics2, keepdims=keepdims)


def generate_l_m_permutations(
    max_band: int,
    name: str = "spherical_harmonics_generate_l_m_permutations") -> Tuple[TensorLike, TensorLike]:  # pylint: disable=line-too-long
  """Generates permutations of degree l and order m for spherical harmonics.

  Args:
    max_band: An integer scalar storing the highest band.
    name: A name for this op. Defaults to
      "spherical_harmonics_generate_l_m_permutations".

  Returns:
    Two tensors of shape `[max_band*max_band]`.
  """
  with tf.name_scope(name):
    degree_l = []
    order_m = []
    for degree in range(0, max_band + 1):
      for order in range(-degree, degree + 1):
        degree_l.append(degree)
        order_m.append(order)
    return (tf.convert_to_tensor(value=degree_l),
            tf.convert_to_tensor(value=order_m))


def generate_l_m_zonal(
    max_band: int,
    name: str = "spherical_harmonics_generate_l_m_zonal") -> Tuple[TensorLike, TensorLike]:  # pylint: disable=line-too-long
  """Generates l and m coefficients for zonal harmonics.

  Args:
    max_band: An integer scalar storing the highest band.
    name: A name for this op. Defaults to
      "spherical_harmonics_generate_l_m_zonal".

  Returns:
    Two tensors of shape `[max_band+1]`, one for degree l and one for order m.
  """
  with tf.name_scope(name):
    degree_l = np.linspace(0, max_band, num=max_band + 1, dtype=np.int32)
    order_m = np.zeros(max_band + 1, dtype=np.int32)
    return (tf.convert_to_tensor(value=degree_l),
            tf.convert_to_tensor(value=order_m))


def _evaluate_legendre_polynomial_pmm_eval(m, x):
  pmm = tf.pow(1.0 - tf.pow(x, 2.0), tf.cast(m, dtype=x.dtype) / 2.0)
  ones = tf.ones_like(m)
  pmm *= tf.cast(
      tf.pow(-ones, m) * math_helpers.double_factorial(2 * m - 1),
      dtype=pmm.dtype)
  return pmm


def _evaluate_legendre_polynomial_loop_cond(x, n, l, m, pmm, pmm1):  # pylint: disable=unused-argument
  return tf.cast(tf.math.count_nonzero(n <= l), tf.bool)


def _evaluate_legendre_polynomial_loop_body(x, n, l, m, pmm, pmm1):
  n_float = tf.cast(n, dtype=x.dtype)
  m_float = tf.cast(m, dtype=x.dtype)
  pmn = (x * (2.0 * n_float - 1.0) * pmm1 - (n_float + m_float - 1) * pmm) / (
      n_float - m_float)
  pmm = tf.where(tf.less_equal(n, l), pmm1, pmm)
  pmm1 = tf.where(tf.less_equal(n, l), pmn, pmm1)
  n += 1
  return x, n, l, m, pmm, pmm1


def _evaluate_legendre_polynomial_loop(x, m, l, pmm, pmm1):
  n = m + 2
  x, n, l, m, pmm, pmm1 = tf.while_loop(
      cond=_evaluate_legendre_polynomial_loop_cond,
      body=_evaluate_legendre_polynomial_loop_body,
      loop_vars=[x, n, l, m, pmm, pmm1])
  return pmm1


def _evaluate_legendre_polynomial_branch(l, m, x, pmm):
  pmm1 = x * (2.0 * tf.cast(m, dtype=x.dtype) + 1.0) * pmm
  # if, l == m + 1 return pmm1, otherwise lift to the next band.
  res = tf.where(
      tf.equal(l, m + 1), pmm1,
      _evaluate_legendre_polynomial_loop(x, m, l, pmm, pmm1))
  return res


def evaluate_legendre_polynomial(degree_l: TensorLike,
                                 order_m: TensorLike,
                                 x: TensorLike) -> TensorLike:
  """Evaluates the Legendre polynomial of degree l and order m at x.

  Note:
    This function is implementing the algorithm described in p. 10 of `Spherical
    Harmonic Lighting: The Gritty Details`.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    degree_l: An integer tensor of shape `[A1, ..., An]` corresponding to the
      degree of the associated Legendre polynomial. Note that `degree_l` must be
      non-negative.
    order_m: An integer tensor of shape `[A1, ..., An]` corresponding to the
      order of the associated Legendre polynomial. Note that `order_m` must
      satisfy `0 <= order_m <= l`.
    x: A tensor of shape `[A1, ..., An]` with values in [-1,1].

  Returns:
    A tensor of shape `[A1, ..., An]` containing the evaluation of the legendre
    polynomial.
  """
  degree_l = tf.convert_to_tensor(value=degree_l)
  order_m = tf.convert_to_tensor(value=order_m)
  x = tf.convert_to_tensor(value=x)

  if not degree_l.dtype.is_integer:
    raise ValueError("`degree_l` must be of an integer type.")
  if not order_m.dtype.is_integer:
    raise ValueError("`order_m` must be of an integer type.")
  shape.compare_batch_dimensions(
      tensors=(degree_l, order_m, x),
      last_axes=-1,
      tensor_names=("degree_l", "order_m", "x"),
      broadcast_compatible=True)
  degree_l = asserts.assert_all_above(degree_l, 0)
  order_m = asserts.assert_all_in_range(order_m, 0, degree_l)
  x = asserts.assert_all_in_range(x, -1.0, 1.0)

  pmm = _evaluate_legendre_polynomial_pmm_eval(order_m, x)
  return tf.where(
      tf.equal(degree_l, order_m), pmm,
      _evaluate_legendre_polynomial_branch(degree_l, order_m, x, pmm))


def _spherical_harmonics_normalization(l, m, var_type=tf.float64):
  l = tf.cast(l, dtype=var_type)
  m = tf.cast(m, dtype=var_type)
  numerator = (2.0 * l + 1.0) * math_helpers.factorial(l - tf.abs(m))
  denominator = 4.0 * np.pi * math_helpers.factorial(l + tf.abs(m))
  return tf.sqrt(numerator / denominator)


# pylint: disable=missing-docstring
def _evaluate_spherical_harmonics_branch(degree,
                                         order,
                                         theta,
                                         phi,
                                         sign_order,
                                         var_type=tf.float64):
  sqrt_2 = tf.constant(1.41421356237, dtype=var_type)
  order_float = tf.cast(order, dtype=var_type)
  tmp = sqrt_2 * _spherical_harmonics_normalization(
      degree, order, var_type) * evaluate_legendre_polynomial(
          degree, order, tf.cos(theta))
  positive = tmp * tf.cos(order_float * phi)
  negative = tmp * tf.sin(order_float * phi)
  return tf.where(tf.greater(sign_order, 0), positive, negative)
  # pylint: enable=missing-docstring


def evaluate_spherical_harmonics(
    degree_l: TensorLike,
    order_m: TensorLike,
    theta: TensorLike,
    phi: TensorLike,
    name: str = "spherical_harmonics_evaluate_spherical_harmonics") -> TensorLike:    # pylint: disable=line-too-long
  """Evaluates a point sample of a Spherical Harmonic basis function.

  Note:
    This function is implementating the algorithm and variable names described
    p. 12 of 'Spherical Harmonic Lighting: The Gritty Details.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    degree_l: An integer tensor of shape `[A1, ..., An, C]`, where the last
      dimension represents the band of the spherical harmonics. Note that
      `degree_l` must be non-negative.
    order_m: An integer tensor of shape `[A1, ..., An, C]`, where the last
      dimension represents the index of the spherical harmonics in the band
      `degree_l`. Note that `order_m` must satisfy `0 <= order_m <= l`.
    theta: A tensor of shape `[A1, ..., An, 1]`. This variable stores the polar
      angle of the sameple. Values of theta must be in [0, pi].
    phi: A tensor of shape `[A1, ..., An, 1]`. This variable stores the
      azimuthal angle of the sameple. Values of phi must be in [0, 2pi].
    name: A name for this op. Defaults to
      "spherical_harmonics_evaluate_spherical_harmonics".

  Returns:
    A tensor of shape `[A1, ..., An, C]` containing the evaluation of each basis
    of the spherical harmonics.

  Raises:
    ValueError: if the shape of `theta` or `phi` is not supported.
    InvalidArgumentError: if at least an element of `l`, `m`, `theta` or `phi`
    is outside the expected range.
  """
  with tf.name_scope(name):
    degree_l = tf.convert_to_tensor(value=degree_l)
    order_m = tf.convert_to_tensor(value=order_m)
    theta = tf.convert_to_tensor(value=theta)
    phi = tf.convert_to_tensor(value=phi)

    if not degree_l.dtype.is_integer:
      raise ValueError("`degree_l` must be of an integer type.")
    if not order_m.dtype.is_integer:
      raise ValueError("`order_m` must be of an integer type.")

    shape.compare_dimensions(
        tensors=(degree_l, order_m),
        axes=-1,
        tensor_names=("degree_l", "order_m"))
    shape.check_static(tensor=phi, tensor_name="phi", has_dim_equals=(-1, 1))
    shape.check_static(
        tensor=theta, tensor_name="theta", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(degree_l, order_m, theta, phi),
        last_axes=-2,
        tensor_names=("degree_l", "order_m", "theta", "phi"),
        broadcast_compatible=False)
    # Checks that tensors contain appropriate data.
    degree_l = asserts.assert_all_above(degree_l, 0)
    order_m = asserts.assert_all_in_range(order_m, -degree_l, degree_l)
    theta = asserts.assert_all_in_range(theta, 0.0, np.pi)
    phi = asserts.assert_all_in_range(phi, 0.0, 2.0 * np.pi)

    var_type = theta.dtype
    sign_m = tf.math.sign(order_m)
    order_m = tf.abs(order_m)
    zeros = tf.zeros_like(order_m)
    result_m_zero = _spherical_harmonics_normalization(
        degree_l, zeros, var_type) * evaluate_legendre_polynomial(
            degree_l, zeros, tf.cos(theta))
    result_branch = _evaluate_spherical_harmonics_branch(
        degree_l, order_m, theta, phi, sign_m, var_type)
    return tf.where(tf.equal(order_m, zeros), result_m_zero, result_branch)


def rotate_zonal_harmonics(
    zonal_coeffs: TensorLike,
    theta: TensorLike,
    phi: TensorLike,
    name: str = "spherical_harmonics_rotate_zonal_harmonics") -> TensorLike:
  """Rotates zonal harmonics.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    zonal_coeffs: A tensor of shape `[C]` storing zonal harmonics coefficients.
    theta: A tensor of shape `[A1, ..., An, 1]` storing polar angles.
    phi: A tensor of shape `[A1, ..., An, 1]` storing azimuthal angles.
    name: A name for this op. Defaults to
      "spherical_harmonics_rotate_zonal_harmonics".

  Returns:
    A tensor of shape `[A1, ..., An, C*C]` storing coefficients of the rotated
    harmonics.

  Raises:
    ValueError: If the shape of `zonal_coeffs`, `theta` or `phi` is not
      supported.
  """
  with tf.name_scope(name):
    zonal_coeffs = tf.convert_to_tensor(value=zonal_coeffs)
    theta = tf.convert_to_tensor(value=theta)
    phi = tf.convert_to_tensor(value=phi)

    shape.check_static(
        tensor=zonal_coeffs, tensor_name="zonal_coeffs", has_rank=1)
    shape.check_static(tensor=phi, tensor_name="phi", has_dim_equals=(-1, 1))
    shape.check_static(
        tensor=theta, tensor_name="theta", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(theta, phi),
        last_axes=-2,
        tensor_names=("theta", "phi"),
        broadcast_compatible=False)

    tiled_zonal_coeffs = tile_zonal_coefficients(zonal_coeffs)
    max_band = zonal_coeffs.shape.as_list()[-1]
    l, m = generate_l_m_permutations(max_band - 1)
    broadcast_shape = theta.shape.as_list()[:-1] + l.shape.as_list()
    l_broadcasted = tf.broadcast_to(l, broadcast_shape)
    m_broadcasted = tf.broadcast_to(m, broadcast_shape)
    n_star = tf.sqrt(4.0 * np.pi / (2.0 * tf.cast(l, dtype=theta.dtype) + 1.0))
    return n_star * tiled_zonal_coeffs * evaluate_spherical_harmonics(
        l_broadcasted, m_broadcasted, theta, phi)


def tile_zonal_coefficients(
    coefficients: TensorLike,
    name: str = "spherical_harmonics_tile_zonal_coefficients") -> TensorLike:
  """Tiles zonal coefficients.

  Zonal Harmonics only contains the harmonics where m=0. This function returns
  these coefficients for -l <= m <= l, where l is the rank of `coefficients`.

  Args:
    coefficients: A tensor of shape `[C]` storing zonal harmonics coefficients.
    name: A name for this op. Defaults to
      "spherical_harmonics_tile_zonal_coefficients".

  Returns:
    A tensor of shape `[C*C]` containing zonal coefficients tiled as
    'regular' spherical harmonics coefficients.

  Raises:
    ValueError: if the shape of `coefficients` is not supported.
  """
  with tf.name_scope(name):
    coefficients = tf.convert_to_tensor(value=coefficients)

    shape.check_static(
        tensor=coefficients, tensor_name="coefficients", has_rank=1)

    return tf.concat([
        coeff * tf.ones(shape=(2 * index + 1,), dtype=coefficients.dtype)
        for index, coeff in enumerate(tf.unstack(coefficients, axis=0))
    ],
                     axis=0)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
