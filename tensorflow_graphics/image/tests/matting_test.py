#Copyright 2018 Google LLC
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
"""Tests for matting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.image import matting
from tensorflow_graphics.util import test_case


class MattingTest(test_case.TestCase):

  def test_laplacian_weights_jacobian_random(self):
    """Tests the Jacobian of the laplacian_weights function."""
    shape = np.random.randint(3, 5, size=3)
    image_init = np.random.uniform(0.0, 1.0, size=shape.tolist() + [3])
    image = tf.convert_to_tensor(value=image_init)

    weights = matting.laplacian_weights(image)

    self.assert_jacobian_is_correct(image, image_init, weights)

  @parameterized.parameters(
      (3, (None, None, None, 1)),
      (3, (None, None, None, 3)),
      (5, (None, None, None, 1)),
      (5, (None, None, None, 3)),
      (3, (1, 3, 3, 1)),
      (3, (1, 3, 3, 3)),
      (5, (1, 5, 5, 1)),
      (5, (1, 5, 5, 3)),
  )
  def test_laplacian_weights_not_raised(self, size, *shapes):
    """Tests that the shape exceptions are not raised."""
    laplacian = lambda image: matting.laplacian_weights(image, size=size)
    self.assert_exception_is_not_raised(laplacian, shapes)

  @parameterized.parameters(
      ("tensor must have a rank of 4, but it has rank", (1,)),
      ("tensor must have a rank of 4, but it has rank", (1, 1, 1, 1, 1)),
  )
  def test_laplacian_weights_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(matting.laplacian_weights, error_msg,
                                    shapes)

  def test_loss_jacobian_random(self):
    """Tests the Jacobian of the matting loss function."""
    shape = np.random.randint(3, 5, size=3)
    matte_init = np.random.uniform(0.0, 1.0, size=shape.tolist() + [1])
    shape[1:3] -= 2
    weights_init = np.random.uniform(0.0, 1.0, size=shape.tolist() + [9, 9])
    matte = tf.convert_to_tensor(value=matte_init)
    weights = tf.convert_to_tensor(value=weights_init)

    loss = matting.loss(matte, weights)

    with self.subTest(name="matte"):
      self.assert_jacobian_is_correct(matte, matte_init, loss)
    with self.subTest(name="weights"):
      self.assert_jacobian_is_correct(weights, weights_init, loss)

  @parameterized.parameters(
      ((None, None, None, 1), (None, None, None, 9, 9)),
      ((None, None, None, 1), (None, None, None, 25, 25)),
      ((1, 6, 6, 1), (1, 4, 4, 9, 9)),
      ((1, 10, 10, 1), (1, 6, 6, 25, 25)),
  )
  def test_loss_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(matting.loss, shapes)

  @parameterized.parameters(
      ("must have exactly 1 dimensions in axis -1", (1, 6, 6, 2),
       (1, 4, 4, 9, 9)),
      ("must have exactly 9 dimensions in axis -2", (1, 6, 6, 1),
       (1, 4, 4, 1, 9)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1),
       (2, 4, 4, 9, 9)),
  )
  def test_loss_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(matting.loss, error_msg, shapes)

  def test_loss_opposite_images(self):
    """Tests that passing opposite images results in a loss close to 0.0."""
    shape = np.random.randint(3, 5, size=3).tolist()
    image = np.random.uniform(0.0, 1.0, size=shape + [1])

    weights = matting.laplacian_weights(image)
    loss = matting.loss(1.0 - image, weights)

    self.assertAllClose(loss, 0.0, atol=1e-4)

  def test_loss_same_images(self):
    """Tests that passing same images results in a loss close to 0.0."""
    shape = np.random.randint(3, 5, size=3).tolist()
    image = np.random.uniform(0.0, 1.0, size=shape + [1])

    weights = matting.laplacian_weights(image)
    loss = matting.loss(image, weights)

    self.assertAllClose(loss, 0.0, atol=1e-4)

  def test_loss_positive(self):
    """Tests that the loss is always greater or equal to 0.0."""
    shape = np.random.randint(3, 5, size=3).tolist()
    image = tf.random.uniform(minval=0.0, maxval=1.0, shape=shape + [3])
    matte = tf.random.uniform(minval=0.0, maxval=1.0, shape=shape + [1])

    weights = matting.laplacian_weights(image)
    loss = matting.loss(matte, weights)

    self.assertAllGreaterEqual(loss, 0.0)


if __name__ == "__main__":
  test_case.main()
