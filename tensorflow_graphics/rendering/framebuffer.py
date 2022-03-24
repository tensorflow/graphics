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
"""Storage classes for framebuffers and related data."""

from typing import Dict, Optional

import dataclasses
import tensorflow as tf


@dataclasses.dataclass
class RasterizedAttribute(object):
  """A single rasterized attribute and optionally its screen-space derivatives.

  Tensors are expected to have shape [batch, height, width, channels] or
  [batch, num_layers, height, width, channels].

  Immutable once created.
  """
  value: tf.Tensor
  d_dx: Optional[tf.Tensor] = None
  d_dy: Optional[tf.Tensor] = None

  def __post_init__(self):
    # Checks that all input tensors have the same shape and rank.
    tensors = [self.value, self.d_dx, self.d_dy]
    shapes = [
        tensor.shape.as_list() for tensor in tensors if tensor is not None
    ]
    ranks = [len(shape) for shape in shapes]
    if not all(rank == ranks[0] for rank in ranks):
      raise ValueError(
          "Expected value and derivatives to be of the same rank, but found"
          f" ranks {shapes}")

    value_rank = len(self.value.shape)
    if value_rank not in (4, 5):
      raise ValueError(
          f"Expected input value to be rank 4 or 5, but is {value_rank}")

    same_as_value = True
    static_shapes = [self.value.shape]
    if self.d_dx is not None:
      same_as_value = tf.logical_and(
          same_as_value, tf.equal(tf.shape(self.value), tf.shape(self.d_dx)))
      static_shapes.append(self.d_dx.shape)
    if self.d_dy is not None:
      same_as_value = tf.logical_and(
          same_as_value, tf.equal(tf.shape(self.value), tf.shape(self.d_dy)))
      static_shapes.append(self.d_dy.shape)
    tf.debugging.assert_equal(
        same_as_value,
        True,
        message="Expected all input shapes to be the same but found: " +
        ", ".join([str(s) for s in static_shapes]))

  @property
  def is_multi_layer(self):
    return len(self.value.shape) == 5

  def layer(self, index):
    """Slices at the given layer index, returning a single-layer attribute."""
    if not self.is_multi_layer:
      if index > 0:
        raise ValueError(
            f"Invalid layer index {index} for single-layer RasterizedAttribute."
        )
      return self

    safe_layer = lambda x: x[:, index, ...] if x is not None else None
    return RasterizedAttribute(
        value=self.value[:, index, ...],
        d_dx=safe_layer(self.d_dx),
        d_dy=safe_layer(self.d_dy))


@dataclasses.dataclass
class Framebuffer(object):
  """A framebuffer holding rasterized values required for deferred shading.

  Tensors are expected to have shape [batch, height, width, channels].

  For now, the fields are specialized for triangle rendering. Other primitives
  may be supported in the future.

  Immutable once created. Uses cached_property to avoid creating redundant
  tf ops when properties are accessed multiple times.
  """
  # The barycentric weights of the pixel centers in the covering triangle.
  barycentrics: RasterizedAttribute
  # The index of the triangle covering this pixel. Not differentiable.
  triangle_id: tf.Tensor
  # The indices of the vertices of the triangle covering this pixel.
  # Not differentiable.
  vertex_ids: tf.Tensor
  # A mask of the pixels covered by a triangle. 1 if covered, 0 if background.
  # Not differentiable.
  foreground_mask: tf.Tensor

  # Other rasterized attribute values (e.g., colors, UVs, normals, etc.).
  attributes: Dict[str, RasterizedAttribute] = dataclasses.field(
      default_factory=dict)

  def __post_init__(self):
    # Checks that all buffers have rank and same shape up to the
    # number of channels.
    values = [self.barycentrics.value, self.triangle_id, self.vertex_ids,
              self.foreground_mask]
    values += [v.value for k, v in self.attributes.items()]

    ranks = [len(v.shape) for v in values]
    shapes = [tf.shape(v) for v in values]
    if not all(rank == ranks[0] for rank in ranks):
      raise ValueError(
          f"Expected all inputs to have the same rank, but found {shapes}")

    same_as_first = [
        tf.reduce_all(tf.equal(shapes[0][:-1], s[:-1])) for s in shapes[1:]
    ]
    all_same_as_first = tf.reduce_all(same_as_first)
    tf.debugging.assert_equal(
        all_same_as_first,
        True,
        message="Expected all input shapes to be the same "
        "(up to channels), but found: " + ", ".join([str(s) for s in shapes]))

  @property
  def batch_size(self):
    return tf.shape(self.triangle_id)[0]

  @property
  def is_multi_layer(self):
    return len(self.triangle_id.shape) == 5

  @property
  def num_layers(self):
    if self.is_multi_layer:  # pylint: disable=using-constant-test
      return tf.shape(self.triangle_id)[1]
    else:
      return 1

  @property
  def height(self):
    axis = 2 if self.is_multi_layer else 1  # pylint: disable=using-constant-test
    return tf.shape(self.triangle_id)[axis]

  @property
  def width(self):
    axis = 3 if self.is_multi_layer else 2  # pylint: disable=using-constant-test
    return tf.shape(self.triangle_id)[axis]

  @property
  def pixel_count(self):
    return self.num_layers * self.height * self.width

  @property
  def background_mask(self):
    return tf.constant(
        1, dtype=self.foreground_mask.dtype) - self.foreground_mask

  def layer(self, index):
    """Slices at the given layer index, returning a single-layer Framebuffer."""
    if not self.is_multi_layer:
      if index > 0:
        raise ValueError(
            f"Invalid layer index {index} for single-layer Framebuffer.")
      return self

    return Framebuffer(
        triangle_id=self.triangle_id[:, index, ...],
        vertex_ids=self.vertex_ids[:, index, ...],
        foreground_mask=self.foreground_mask[:, index, ...],
        attributes={k: v.layer(index) for k, v in self.attributes.items()},
        barycentrics=self.barycentrics.layer(index))
