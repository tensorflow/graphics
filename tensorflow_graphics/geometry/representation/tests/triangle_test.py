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
"""Tests for triangle."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.geometry.representation import triangle
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import test_case


class TriangleTest(test_case.TestCase):  # pylint: disable=missing-class-docstring

  @parameterized.parameters(
      ((0., 0., 0.), (0., 0., 0.), (0., 0., 0.)),
      ((1., 0., 0.), (0., 0., 0.), (0., 0., 0.)),
      ((0., 0., 0.), (0., 1., 0.), (0., 0., 0.)),
      ((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)),
  )
  def test_normal_assert(self, v0, v1, v2):
    """Tests the triangle normal assertion."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(triangle.normal(v0, v1, v2))

  @parameterized.parameters(
      ((3,), (3,), (3,)),
      ((1, 3), (2, 3), (2, 3)),
      ((2, 3), (1, 3), (2, 3)),
      ((2, 3), (2, 3), (1, 3)),
      ((None, 3), (None, 3), (None, 3)),
  )
  def test_normal_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(triangle.normal, shapes)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (3, 3),
       (2, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (2, 3),
       (3, 3)),
      ("must have exactly 3 dimensions in axis -1", (1,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (4,)),
  )
  def test_normal_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(triangle.normal, error_msg, shapes)

  @parameterized.parameters(
      ((0., 0., 1.), (0., 0., 0.), (0., 1., 0.)),
      ((0., 1., 0.), (0., 0., 0.), (0., 0., 1.)),
      ((1., 0., 0.), (0., 0., 0.), (0., 0., 1.)),
      ((0., 0., 1.), (0., 0., 0.), (1., 0., 0.)),
      ((0., 1., 0.), (0., 0., 0.), (1., 0., 0.)),
      ((1., 0., 0.), (0., 0., 0.), (0., 1., 0.)),
      ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
  )
  def test_normal_jacobian_preset(self, *vertices):
    """Test the Jacobian of the triangle normal function."""
    v0_init, v1_init, v2_init = [np.array(v) for v in vertices]
    v0_tensor, v1_tensor, v2_tensor = [
        tf.convert_to_tensor(value=v) for v in [v0_init, v1_init, v2_init]
    ]

    with self.subTest(name="v0"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.normal(x, v1_tensor, v2_tensor), [v0_init])

    with self.subTest(name="v1"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.normal(v0_tensor, x, v2_tensor), [v1_init])

    with self.subTest(name="v2"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.normal(v0_tensor, v1_tensor, x), [v2_init])

  def test_normal_jacobian_random(self):
    """Test the Jacobian of the triangle normal function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    v0_init, v1_init, v2_init = [
        np.random.random(size=tensor_shape + [3]) for _ in range(3)
    ]
    v0_tensor, v1_tensor, v2_tensor = [
        tf.convert_to_tensor(value=v) for v in [v0_init, v1_init, v2_init]
    ]

    with self.subTest(name="v0"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.normal(x, v1_tensor, v2_tensor), [v0_init],
          atol=1e-4,
          delta=1e-9)

    with self.subTest(name="v1"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.normal(v0_tensor, x, v2_tensor), [v1_init],
          atol=1e-4,
          delta=1e-9)

    with self.subTest(name="v2"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.normal(v0_tensor, v1_tensor, x), [v2_init],
          atol=1e-4,
          delta=1e-9)

  @parameterized.parameters(
      (((0., 0., 1.), (0., 0., 0.), (0., 1., 0.)), ((-1., 0., 0.),)),
      (((0., 1., 0.), (0., 0., 0.), (0., 0., 1.)), ((1., 0., 0.),)),
      (((1., 0., 0.), (0., 0., 0.), (0., 0., 1.)), ((0., -1., 0.),)),
      (((0., 0., 1.), (0., 0., 0.), (1., 0., 0.)), ((0., 1., 0.),)),
      (((0., 1., 0.), (0., 0., 0.), (1., 0., 0.)), ((0., 0., -1.),)),
      (((1., 0., 0.), (0., 0., 0.), (0., 1., 0.)), ((0., 0., 1.),)),
      (((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), ((-np.sqrt(1. / 3.),) * 3,)),
  )
  def test_normal_preset(self, test_inputs, test_outputs):
    """Tests the triangle normal computation."""
    self.assert_output_is_correct(triangle.normal, test_inputs, test_outputs)

  @parameterized.parameters((False,), (True,))
  def test_normal_random(self, clockwise):
    """Tests the triangle normal computation in each axis."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    zeros = np.zeros(shape=tensor_shape + [1])

    for i in range(3):
      v0 = np.random.random(size=tensor_shape + [3])
      v1 = np.random.random(size=tensor_shape + [3])
      v2 = np.random.random(size=tensor_shape + [3])
      v0[..., i] = 0.
      v1[..., i] = 0.
      v2[..., i] = 0.
      n = np.zeros_like(v0)
      n[..., i] = 1.
      normal = triangle.normal(v0, v1, v2, clockwise)

      with self.subTest(name="n"):
        self.assertAllClose(tf.abs(normal), n)

      with self.subTest(name="v1-v0"):
        self.assertAllClose(vector.dot(normal, (v1 - v0)), zeros)

      with self.subTest(name="v2-v0"):
        self.assertAllClose(vector.dot(normal, (v2 - v0)), zeros)

  @parameterized.parameters(
      ((3,), (3,), (3,)),
      ((1, 3), (2, 3), (2, 3)),
      ((2, 3), (1, 3), (2, 3)),
      ((2, 3), (2, 3), (1, 3)),
      ((None, 3), (None, 3), (None, 3)),
  )
  def test_area_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(triangle.area, shapes)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (3, 3),
       (2, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (2, 3),
       (3, 3)),
      ("must have exactly 3 dimensions in axis -1", (1,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (4,)),
  )
  def test_area_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(triangle.area, error_msg, shapes)

  @parameterized.parameters(
      ((0., 0., 0.), (0., 1e-6, 0.), (1e-6, 0., 0.)),
      ((0., 0., 0.), (0., 1e-6, 0.), (1., 0., 0.)),
      ((0., 0., 1.), (0., 0., 0.), (0., 1., 0.)),
      ((0., 1., 0.), (0., 0., 0.), (0., 0., 1.)),
      ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
  )
  def test_area_jacobian_preset(self, *vertices):
    """Test the Jacobian of the triangle area function.

    Args:
      *vertices: List of 3 tuples of shape [3] denoting 3 vertex coordinates.
    Note: 'almost' degenerate triangles are tested instead of completely
      degenrate triangles, as the gradient of tf.Norm is NaN at 0.
    """
    v0_init, v1_init, v2_init = [np.array(v) for v in vertices]
    v0_tensor, v1_tensor, v2_tensor = [
        tf.convert_to_tensor(value=v) for v in [v0_init, v1_init, v2_init]
    ]

    with self.subTest(name="v0"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.area(x, v1_tensor, v2_tensor), [v0_init])

    with self.subTest(name="v1"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.area(v0_tensor, x, v2_tensor), [v1_init])

    with self.subTest(name="v2"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.area(v0_tensor, v1_tensor, x), [v2_init])

  def test_area_jacobian_random(self):
    """Test the Jacobian of the triangle normal function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    v0_init, v1_init, v2_init = [
        np.random.random(size=tensor_shape + [3]) for _ in range(3)
    ]
    v0_tensor, v1_tensor, v2_tensor = [
        tf.convert_to_tensor(value=v) for v in [v0_init, v1_init, v2_init]
    ]

    with self.subTest(name="v0"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.area(x, v1_tensor, v2_tensor), [v0_init],
          atol=1e-4,
          delta=1e-9)

    with self.subTest(name="v1"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.area(v0_tensor, x, v2_tensor), [v1_init],
          atol=1e-4,
          delta=1e-9)

    with self.subTest(name="v2"):
      self.assert_jacobian_is_correct_fn(
          lambda x: triangle.area(v0_tensor, v1_tensor, x), [v2_init],
          atol=1e-4,
          delta=1e-9)

  @parameterized.parameters(
      (((0., 0., 0.), (0., 0., 0.), (0., 0., 0.)), ((0.,),)),
      (((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)), ((0.,),)),
      (((0., 0., 1.), (0., 0., 0.), (0., 1., 0.)), ((0.5,),)),
      (((0., 1., 0.), (0., 0., 0.), (0., 0., 1.)), ((0.5,),)),
      (((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), ((np.sqrt(3. / 4.),),)),
  )
  def test_area_preset(self, test_inputs, test_outputs):
    """Tests the triangle area computation."""
    self.assert_output_is_correct(triangle.area, test_inputs, test_outputs)


if __name__ == "__main__":
  test_case.main()
