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
"""Tests for shape utility functions."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.util import shape
from tensorflow_graphics.util import test_case


class ShapeTest(test_case.TestCase):

  @parameterized.parameters(
      (None, None, False),
      ((2, 3, 5, 7), (2, 3, 5, 7), True),
      ((1, 3, 5, 7), (2, 3, 5, 7), True),
      ((2, 3, 5, 7), (1, 3, 5, 7), True),
      ((None, 3, 5, 7), (None, 3, 5, 7), True),
      ((2, None, 5, 7), (2, 3, None, 7), True),
      ((1, 3, None, 7), (2, 3, 5, None), True),
      ((None, 3, 5, 7), (None, 3, 5, 7), True),
      ((None, 3, 5, 7), (None, 3, 5, 1), True),
      ((None, 3, 5, 7), (None, 3, 1, 1), True),
      ((None, 3, 5, 7), (None, 1, 1, 1), True),
      ((None, 3, 5, 7), (2, 3, 5, 7), True),
      ((None, 3, 5, 7), (1, 3, 5, 7), True),
      ((None, 3, 5, 7), (None, 5, 7), True),
      ((None, 3, 5, 7), (None, 7), True),
      ((None, 3, 5, 7), (None, 1, 1), True),
      ((None, 3, 5, 7), (None, 1), True),
      ((None, 3, 5, 7), (None,), True),
      ((2, 3, 5, 7), (3, 3, 5, 7), False),
      ((3, 3, 5, 7), (2, 3, 5, 7), False),
      ((None, 3, 5, 7), (None, 3, 5), False),
      ((None, 3, 5, 7), (None, 5), False),
      ((None, 3, 5, 7), (None, 3, 5, 7, 1), False),
      ((None, 3, 5, 7), (None, 2, 5, 7), False),
      ((None, 3, 5, 7), (None, 3, 4, 7), False),
      ((None, 3, 5, 7), (None, 3, 5, 6), False),
  )
  def test_is_broadcast_compatible(self, shape_x, shape_y, broadcastable):
    """Checks if the is_broadcast_compatible function works as expected."""
    if tf.executing_eagerly():
      if (shape_x is None or shape_y is None or None in shape_x or
          None in shape_y):
        return
      shape_x = tf.compat.v1.placeholder_with_default(
          tf.zeros(shape_x, dtype=tf.float32), shape=shape_x).shape
      shape_y = tf.compat.v1.placeholder_with_default(
          tf.zeros(shape_y, dtype=tf.float32), shape=shape_y).shape
    else:
      shape_x = tf.compat.v1.placeholder(shape=shape_x, dtype=tf.float32).shape
      shape_y = tf.compat.v1.placeholder(shape=shape_y, dtype=tf.float32).shape

    self.assertEqual(
        shape.is_broadcast_compatible(shape_x, shape_y), broadcastable)

  @parameterized.parameters(
      (None, None, None),
      ((2, 3, 5, 7), (2, 3, 5, 7), (2, 3, 5, 7)),
      ((1, 3, 5, 7), (2, 3, 5, 7), (2, 3, 5, 7)),
      ((2, 3, 5, 7), (1, 3, 5, 7), (2, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 3, 5, 7), (None, 3, 5, 7)),
      ((2, None, 5, 7), (2, 3, None, 7), (2, 3, 5, 7)),
      ((1, 3, None, 7), (2, 3, 5, None), (2, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 3, 5, 7), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 3, 5, 1), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 3, 1, 1), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 1, 1, 1), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (2, 3, 5, 7), (2, 3, 5, 7)),
      ((None, 3, 5, 7), (1, 3, 5, 7), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 5, 7), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 7), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 1, 1), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None, 1), (None, 3, 5, 7)),
      ((None, 3, 5, 7), (None,), (None, 3, 5, 7)),
  )
  def test_get_broadcasted_shape(self, shape_x, shape_y, broadcasted_shape):
    """Checks if the get_broadcasted_shape function works as expected."""
    if tf.executing_eagerly():
      if (shape_x is None or shape_y is None or None in shape_x or
          None in shape_y):
        return
      shape_x = tf.compat.v1.placeholder_with_default(
          tf.zeros(shape_x, dtype=tf.float32), shape=shape_x).shape
      shape_y = tf.compat.v1.placeholder_with_default(
          tf.zeros(shape_y, dtype=tf.float32), shape=shape_y).shape
    else:
      shape_x = tf.compat.v1.placeholder(shape=shape_x, dtype=tf.float32).shape
      shape_y = tf.compat.v1.placeholder(shape=shape_y, dtype=tf.float32).shape

    self.assertAllEqual(
        shape.get_broadcasted_shape(shape_x, shape_y), broadcasted_shape)

  @parameterized.parameters(
      ("has_rank must be of type int", "a", None, None, None),
      ("has_rank_greater_than must be of type int", 3, "a", None, None),
      ("has_rank_less_than must be of type int", None, None, "a", None),
      ("must have a rank of 1, but it has", 1, None, None, None),
      ("must have a rank greater than 3, but it has", None, 3, None, None),
      ("must have a rank less than 3, but it has", None, None, 3, None),
      ("has_dim_equals must be of type list or tuple", None, None, None, 0),
      ("has_dim_equals must consist of axis-value pairs", None, None, None,
       (0,)),
      ("has_dim_equals must consist of axis-value pairs", None, None, None,
       ((0, 0, 0),)),
      ("must have exactly 2 dimensions in axis 0", None, None, None, ((0, 2),)),
      ("must have exactly 3 dimensions in axis 1", None, None, None, ((0, 1),
                                                                      (1, 3))),
      ("must have greater than 2 dimensions in axis 1", None, None, None, None,
       ((1, 2),)),
      ("must have greater than 3 dimensions in axis 1", None, None, None, None,
       ((1, 3),)),
      ("must have less than 3 dimensions in axis 2", None, None, None, None,
       None, ((2, 3),)),
      ("must have less than 2 dimensions in axis -1", None, None, None, None,
       None, ((-1, 2),)),
  )
  def test_check_static_raises_exceptions(self,
                                          error_msg,
                                          has_rank,
                                          has_rank_greater_than,
                                          has_rank_less_than,
                                          has_dim_equals,
                                          has_dim_greater_than=None,
                                          has_dim_less_than=None):
    """Tests that check_static raises expected exceptions."""
    self.assert_exception_is_raised(
        shape.check_static,
        error_msg,
        shapes=((1, 2, 3),),
        has_rank=has_rank,
        has_rank_greater_than=has_rank_greater_than,
        has_rank_less_than=has_rank_less_than,
        has_dim_equals=has_dim_equals,
        has_dim_greater_than=has_dim_greater_than,
        has_dim_less_than=has_dim_less_than)

  @parameterized.parameters(
      (((),), 0, None, 1, None),
      (((1,),), 1, 0, 2, (0, 1)),
      (((1, 2),), 2, 1, 3, (0, 1)),
      (((1, 2),), 2, 1, 3, ((0, 1), (1, 2))),
      (((1, 2, 3),), 3, 2, 4, (0, 1)),
      (((1, 2, 3),), 3, 2, 4, ((0, 1), (1, 2), (2, 3))),
  )
  def test_check_static_raises_no_exceptions(self, shapes, has_rank,
                                             has_rank_greater_than,
                                             has_rank_less_than,
                                             has_dim_equals):
    """Tests that check_static works for various inputs."""
    self.assert_exception_is_not_raised(
        shape.check_static,
        shapes=shapes,
        has_rank=has_rank,
        has_rank_greater_than=has_rank_greater_than,
        has_rank_less_than=has_rank_less_than,
        has_dim_equals=has_dim_equals)

  @parameterized.parameters(
      ("tensors must be of type list or tuple", [], (-1), False),
      ("At least 2 tensors are required.", ((1,),), (-1), False),
      ("tensors and last_axes must have the same length", ((1, 1), (1, 1)),
       (0, 0, 0), False),
      ("Some axes are out of bounds.", ((1, 2), (2, 2), (3, 2)),
       (1, 1, 2), False),
      ("Not all batch dimensions are identical.", ((1, 2, 3), (2, 2, 3)),
       (0, 0), False),
      ("Not all batch dimensions are identical.", ((1, 2, 3), (2, 2, 3)),
       (2, 2), False),
      ("Some axes are out of bounds.", ((1, 2), (1, 2), (2, 2)),
       (2, 2, 2), False),
      ("Not all batch dimensions are broadcast-compatible.",
       ((1, 2), (2, 2), (3, 2)), (1, 1, 1), True),
      ("Not all batch dimensions are identical.", ((1, 2, 3), (5, 2, 4),
                                                   (5, 1, 5)), 1, False),
      ("Not all batch dimensions are identical.", ((None, 2, 3), (5, None, 4),
                                                   (5, 1, None)), 1, False),
      ("Not all batch dimensions are identical.",
       ((None, 2, 3), (5, None, 4), (5, 2, None)), (1, 1, 2), False),
      ("Not all batch dimensions are identical.",
       ((1, 2, 3), (1, 2, 3), (1, 2, 3)), (1, 1, 2), False),
      ("Not all batch dimensions are broadcast-compatible.",
       ((None, 2, 3), (5, None, 4), (5, 3, None)), 1, True),
  )
  def test_compare_batch_dimensions_raises_exceptions(self, error_msg,
                                                      tensor_shapes, last_axes,
                                                      broadcast_compatible):
    """Tests that compare_batch_dimensions raises expected exceptions."""
    if not tensor_shapes:
      tensors = 0
    else:
      if all(shape.is_static(tensor_shape) for tensor_shape in tensor_shapes):
        tensors = [tf.ones(tensor_shape) for tensor_shape in tensor_shapes]
      else:
        # Dynamic shapes are not supported in eager mode.
        if tf.executing_eagerly():
          return
        tensors = [
            tf.compat.v1.placeholder(shape=tensor_shape, dtype=tf.float32)
            for tensor_shape in tensor_shapes
        ]
    self.assert_exception_is_raised(
        shape.compare_batch_dimensions,
        error_msg,
        shapes=[],
        tensors=tensors,
        last_axes=last_axes,
        broadcast_compatible=broadcast_compatible)

  @parameterized.parameters(
      (((1, 1), (1, 2)), 0, False, 0),
      (((1, 1), (1, 2)), (0, 0), True, 0),
      (((4, 2, 3), (4, 2, 5, 6)), (1, 1), False, (0, 0)),
      (((4, 2, 3), (1, 2, 5, 6)), 1, True, (0, 0)),
      (((4, 2, 3), (2, 5, 6)), (1, 0), True, 0),
      (((1, 2), (1, 2), (2, 2)), (1, 1, 1), True, 0),
      (((1, 2, 3), (5, 2, 3), (3,)), (2, 2, 0), True, (0, 1, 0)),
      (((1, 2, 3), (5, 2, 4), (5, 1, 5)), 1, True, 0),
      (((None, 2, 3), (5, None, 4), (5, 1, None)), 1, True, 0),
      (((None, 2, 3), (None, 2, 4), (None, 2, None)), 1, False, 0),
      (((None, 2, 3), (None, 2, 4), (None, 2, None)), 1, True, 0),
      (((None, None, 3), (None, None, 4), (None, None, None)), 1, False, 0),
      (((None, None, 3), (None, None, 4), (None, 2, None)), 1, False, 0),
      (((2, None, 3), (2, None, 4), (None, 2, None)), 1, False, 0),
  )
  def test_compare_batch_dimensions_raises_no_exceptions(
      self, tensor_shapes, last_axes, broadcast_compatible, initial_axes):
    """Tests that compare_batch_dimensions works for various inputs."""
    if all(shape.is_static(tensor_shape) for tensor_shape in tensor_shapes):
      tensors = [tf.ones(tensor_shape) for tensor_shape in tensor_shapes]
    else:
      # Dynamic shapes are not supported in eager mode.
      if tf.executing_eagerly():
        return
      tensors = [
          tf.compat.v1.placeholder(shape=tensor_shape, dtype=tf.float32)
          for tensor_shape in tensor_shapes
      ]
    self.assert_exception_is_not_raised(
        shape.compare_batch_dimensions,
        shapes=[],
        tensors=tensors,
        last_axes=last_axes,
        broadcast_compatible=broadcast_compatible,
        initial_axes=initial_axes)

  @parameterized.parameters(
      ("tensors must be of type list or tuple", [], (-1,)),
      ("At least 2 tensors are required.", (1,), (-1,)),
      ("tensors and axes must have the same length", (1, 1), (0, 0, 0)),
      ("Some axes are out of bounds.", ((1, 2), (2, 2), (3, 2)), (1, 1, 2)),
      ("Some axes are out of bounds.", ((1, 2), (2, 2), (3, 2)), (-2, -2, -3)),
      ("must have the same number of dimensions", ((1, 2, 3), (2, 2, 3)),
       (0, 0)),
      ("must have the same number of dimensions", ((1, 2, 5), (2, 2, 3)),
       (1, -1)),
      ("must have the same number of dimensions", ((1, 2), (1, 2), (2, 2)),
       (0, 0, 1)),
      ("must have the same number of dimensions", ((1, 2), (2, 2), (3, 2)),
       (-1, -2, 0)),
  )
  def test_compare_dimensions_raises_exceptions(self, error_msg, tensor_shapes,
                                                axes):
    """Tests that compare_dimensions raises expected exceptions."""
    if not tensor_shapes:
      tensors = 0
    else:
      tensors = [tf.ones(tensor_shape) for tensor_shape in tensor_shapes]
    self.assert_exception_is_raised(
        shape.compare_dimensions,
        error_msg,
        shapes=[],
        tensors=tensors,
        axes=axes)

  @parameterized.parameters(
      ((1, 1), (0, 0)),
      ((1, 1), 0),
      (((4, 2, 3), (4, 2, 5, 6)), (1, 1)),
      (((4, 2, 3), (1, 2, 5, 6)), (1, 1)),
      (((4, 2, 3), (2, 5, 6)), (1, 0)),
      (((1, 2), (1, 2), (2, 2)), (1, 1, 0)),
      (((1, 2, 3), (2, 3), (3,)), (2, 1, 0)),
  )
  def test_compare_dimensions_raises_no_exceptions(self, tensor_shapes, axes):
    """Tests that compare_dimensions works for various inputs."""
    tensors = [tf.ones(tensor_shape) for tensor_shape in tensor_shapes]
    self.assert_exception_is_not_raised(
        shape.compare_dimensions, shapes=[], tensors=tensors, axes=axes)

  @parameterized.parameters(
      ((1,), True),
      ((1, 2), True),
      ((None,), False),
      ((None, 2), False),
  )
  def test_is_static(self, tensor_shape, is_static):
    """Tests that is_static correctly checks if shape is static."""
    if tf.executing_eagerly():
      return
    tensor = tf.compat.v1.placeholder(shape=tensor_shape, dtype=tf.float32)

    with self.subTest(name="tensor_shape_is_list"):
      self.assertEqual(shape.is_static(tensor_shape), is_static)

    with self.subTest(name="tensor_shape"):
      self.assertEqual(shape.is_static(tensor.shape), is_static)


if __name__ == "__main__":
  test_case.main()
