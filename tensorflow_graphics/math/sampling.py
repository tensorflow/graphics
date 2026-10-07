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
"""This module implements different 1D sampling strategies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape
from tensorflow_graphics.util.type_alias import TensorLike


def regular_1d(near: TensorLike,
               far: TensorLike,
               num_samples: int,
               name="regular_1d") -> tf.Tensor:
  """Sample evenly spaced numbers on an interval.

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    name:  A name for this op that defaults to "regular_1d".

  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)

    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)

    return tf.linspace(near, far, num_samples, axis=-1)


def regular_inverse_1d(near: TensorLike,
                       far: TensorLike,
                       num_samples: int,
                       name="regular_inverse_1d") -> tf.Tensor:
  """Sample inverse evenly spaced numbers on an interval.

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    name:  A name for this op that defaults to "regular_inverse_1d".

  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)

    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)

    return 1. / tf.linspace(1. / near, 1. / far, num_samples, axis=-1)


def uniform_1d(near: TensorLike,
               far: TensorLike,
               num_samples: int,
               name="uniform_1d") -> tf.Tensor:
  """Sample uniformly numbers on an interval, with the numbers being sorted.

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    name:  A name for this op that defaults to "uniform_1d".

  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)

    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)

    target_shape = tf.concat([tf.shape(near), [num_samples]], axis=-1)
    random_samples = tf.random.uniform(target_shape,
                                       minval=tf.expand_dims(near, -1),
                                       maxval=tf.expand_dims(far, -1))
    return tf.sort(random_samples, axis=-1)


def stratified_1d(near: TensorLike,
                  far: TensorLike,
                  num_samples: int,
                  name="stratified_1d") -> tf.Tensor:
  """Stratified sampling based on evenly-spaced bin size of an interval.

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    name:  A name for this op that defaults to "stratified_1d".


  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)

    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)

    bin_borders = tf.linspace(0.0, 1.0, num_samples + 1, axis=-1)
    bin_below = bin_borders[..., :-1]
    bin_above = bin_borders[..., 1:]
    target_shape = tf.concat([tf.shape(near), [num_samples]], axis=-1)
    random_point_in_bin = tf.random.uniform(target_shape)
    z_values = bin_below + (bin_above - bin_below) * random_point_in_bin
    z_values = (tf.expand_dims(near, -1) * (1. - z_values)
                + tf.expand_dims(far, -1) * z_values)
    return z_values


def logspace_1d(near: TensorLike,
                far: TensorLike,
                num_samples: int,
                base: float = 10.0,
                name="logspace_1d") -> tf.Tensor:
  """Sample evenly spaced numbers from an interval on a log scale.

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    base: The logarithmic base.
    name:  A name for this op that defaults to "logspace_1d".

  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)
    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)

    linspace = tf.linspace(near, far, num_samples, axis=-1)
    return tf.math.pow(base, linspace)


def _log10(x):
  return tf.math.log(x) / tf.math.log(10.0)


def geomspace_1d(near: TensorLike,
                 far: TensorLike,
                 num_samples: int,
                 name="geomspace_1d") -> tf.Tensor:
  """Sample evenly spaced numbers on a geometric progression (log scale).

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    name:  A name for this op that defaults to "geomspace_1d".

  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)
    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)
    return logspace_1d(_log10(near), _log10(far), num_samples)


def stratified_geomspace_1d(near: TensorLike,
                            far: TensorLike,
                            num_samples: int,
                            name="stratified_geomspace_1d") -> tf.Tensor:
  """Stratified sampling based on evenly-spaced bins on a geometric progression.

  Args:
    near: A tensor of shape `[A1, ... An]` containing the starting points of
      the sampling interval.
    far: A tensor of shape `[A1, ... An]` containing the ending points of
      the sampling interval.
    num_samples: The number M of points to be sampled.
    name:  A name for this op that defaults to "stratified".


  Returns:
    A tensor of shape `[A1, ..., An, M]` indicating the M points on the ray
  """
  with tf.name_scope(name):
    near = tf.convert_to_tensor(near)
    far = tf.convert_to_tensor(far)

    shape.compare_batch_dimensions(
        tensors=(tf.expand_dims(near, axis=-1), tf.expand_dims(far, axis=-1)),
        tensor_names=("near", "far"),
        last_axes=-1,
        broadcast_compatible=True)

    bin_borders = geomspace_1d(near, far, num_samples + 1)
    bin_below = bin_borders[..., :-1]
    bin_above = bin_borders[..., 1:]
    target_shape = tf.concat([tf.shape(near), [num_samples]], axis=-1)
    random_point_in_bin = tf.random.uniform(target_shape)
    z_values = bin_below + (bin_above - bin_below) * random_point_in_bin
    return z_values


def _normalize_pdf(pdf: TensorLike, name="normalize_pdf") -> tf.Tensor:
  """Normalizes a probability density function.

  Args:
    pdf: A tensor of shape `[A1, ..., An, M]` containing the probability
      distribution in M bins.
    name:  A name for this op that defaults to "_normalize_pdf".

  Returns:
    A tensor of shape `[A1, ..., An, M]`.
  """
  with tf.name_scope(name):
    pdf = tf.convert_to_tensor(value=pdf)
    pdf += 1e-5
    return safe_ops.safe_signed_div(pdf, tf.reduce_sum(pdf, -1, keepdims=True))


def _get_cdf(pdf: TensorLike, name="get_cdf"):
  """Estimates the cumulative distribution function of a probability distribution.

  Args:
    pdf: A tensor of shape `[A1, ..., An, M]` containing the probability
      distribution in M bins.
    name:  A name for this op that defaults to "_get_cdf".

  Returns:
    A tensor of shape `[A1, ..., An, M+1]`.
  """
  with tf.name_scope(name):
    pdf = tf.convert_to_tensor(value=pdf)
    batch_shape = tf.shape(pdf)[:-1]
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros(tf.concat([batch_shape, [1]], axis=-1)), cdf], -1)
    return cdf


def inverse_transform_sampling_1d(
    bins: TensorLike,
    pdf: TensorLike,
    num_samples: int,
    name="inverse_transform_sampling_1d") -> tf.Tensor:
  """Sampling 1D points from a distribution using the inverse transform.

     The target distrubution is defined by its probability density function and
     the spatial 1D location of its bins. The new random samples correspond to
     the centers of the bins.

  Args:
    bins: A tensor of shape `[A1, ..., An, M]` containing 1D location of M bins.
      For example, a tensor [a, b, c, d] corresponds to
      the bin structure |--a--|-b-|--c--|d|.
    pdf: A tensor of shape `[A1, ..., An, M]` containing the probability
      distribution in M bins.
    num_samples: The number N of new samples.
    name:  A name for this op that defaults to "inverse_transform_sampling_1d".

  Returns:
    A tensor of shape `[A1, ..., An, N]` indicating the new N random points.
  """

  with tf.name_scope(name):
    bins = tf.convert_to_tensor(value=bins)
    pdf = tf.convert_to_tensor(value=pdf)

    shape.check_static(
        tensor=bins,
        tensor_name="bins",
        has_rank_greater_than=0)
    shape.check_static(
        tensor=pdf,
        tensor_name="pdf",
        has_rank_greater_than=0)
    shape.compare_batch_dimensions(
        tensors=(bins, pdf),
        tensor_names=("bins", "pdf"),
        last_axes=-2,
        broadcast_compatible=True)
    shape.compare_dimensions(
        tensors=(bins, pdf),
        tensor_names=("bins", "pdf"),
        axes=-1)
    # Do not consider the last bin, as the cdf contains has +1 dimension.
    pdf = _normalize_pdf(pdf[..., :-1])
    cdf = _get_cdf(pdf)
    batch_shape = tf.shape(pdf)[:-1]
    # TODO(krematas): Use dynamic values
    batch_dims = tf.get_static_value(tf.rank(pdf) - 1)
    target_shape = tf.concat([batch_shape, [num_samples]], axis=-1)
    uniform_samples = tf.random.uniform(target_shape)
    bin_indices = tf.searchsorted(cdf, uniform_samples, side="right")
    bin_indices = tf.maximum(0, bin_indices - 1)
    z_values = tf.gather(bins, bin_indices, axis=-1, batch_dims=batch_dims)
    return z_values


def inverse_transform_stratified_1d(bin_start: TensorLike,
                                    bin_width: TensorLike,
                                    pdf: TensorLike,
                                    num_samples: int,
                                    name="inverse_transform_stratified_1d"):
  """Stratified sampling 1D points from a distribution using the inverse transform.

    The target distrubution is defined by its probability density function and
    the spatial 1D location of its bins (start and width of each bin).
    The new samples can be sampled from anywhere inside the bin, unlike
    inverse_transform_sampling_1d that returns the selected bin location.

  Args:
    bin_start: A tensor of shape `[A1, ..., An, M]` containing starting position
      of M bins.
    bin_width: A tensor of shape `[A1, ..., An, M]` containing the width of
      M bins.
    pdf: A tensor of shape `[A1, ..., An, M]` containing the probability
      distribution in M bins.
    num_samples: The number N of new samples.
    name:  A name for this op that defaults to "inverse_transform_stratified".

  Returns:
    A tensor of shape `[A1, ..., An, N]` indicating the N points on the ray
  """

  with tf.name_scope(name):
    bin_start = tf.convert_to_tensor(value=bin_start)
    bin_width = tf.convert_to_tensor(value=bin_width)
    pdf = tf.convert_to_tensor(value=pdf)

    shape.check_static(
        tensor=bin_start,
        tensor_name="bin_start",
        has_rank_greater_than=0)
    shape.check_static(
        tensor=bin_width,
        tensor_name="bin_width",
        has_rank_greater_than=0)
    shape.check_static(
        tensor=pdf,
        tensor_name="pdf",
        has_rank_greater_than=0)
    shape.compare_batch_dimensions(
        tensors=(bin_start, pdf, bin_width),
        tensor_names=("bins", "pdf", "bin_width"),
        last_axes=-2,
        broadcast_compatible=True)
    shape.compare_dimensions(
        tensors=(bin_start, pdf, bin_width),
        tensor_names=("bins", "pdf", "bin_width"),
        axes=-1)
    # Do not consider the last bin, as the cdf contains has +1 dimension.
    pdf = _normalize_pdf(pdf[..., :-1])
    cdf = _get_cdf(pdf)
    batch_shape = tf.shape(pdf)[:-1]
    batch_dims = batch_shape.get_shape().as_list()[0]
    target_shape = tf.concat([batch_shape, [num_samples]], axis=-1)
    uniform_samples = tf.random.uniform(target_shape)
    bin_indices = tf.searchsorted(cdf, uniform_samples, side="right")
    below_bin_id = tf.maximum(0, bin_indices - 1)
    above_bin_id = tf.minimum(cdf.shape[-1] - 1, bin_indices)
    below_bin_cdf = tf.gather(cdf, below_bin_id, axis=-1, batch_dims=batch_dims)
    above_bin_cdf = tf.gather(cdf, above_bin_id, axis=-1, batch_dims=batch_dims)
    bin_prob = above_bin_cdf - below_bin_cdf
    bin_prob = tf.where(bin_prob < 1e-5, tf.ones_like(bin_prob), bin_prob)
    below_bin = tf.gather(bin_start, below_bin_id, axis=-1,
                          batch_dims=batch_dims)
    bin_width = tf.gather(bin_width, below_bin_id, axis=-1,
                          batch_dims=batch_dims)
    return below_bin + (uniform_samples - below_bin_cdf) / bin_prob * bin_width

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
