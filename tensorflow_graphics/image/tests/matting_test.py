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
"""Tests for matting."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.image import matting
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import test_case


def _laplacian_matrix(image, size=3, eps=1e-5, name=None):
  """Generates the closed form matting Laplacian matrices.

  Generates the closed form matting Laplacian as proposed by Levin et
  al. in "A Closed Form Solution to Natural Image Matting".

  Args:
    image: A tensor of shape `[B, H, W, C]`.
    size: An `int` representing the size of the patches used to enforce
      smoothness.
    eps: A small number of type `float` to regularize the problem.
    name: A name for this op. Defaults to "matting_laplacian_matrix".

  Returns:
    A tensor of shape `[B, H, W, size^2, size^2]` containing the
    matting Laplacian matrices.

  Raises:
    ValueError: If `image` is not of rank 4.
  """
  with tf.compat.v1.name_scope(name, "matting_laplacian_matrix", [image]):
    image = tf.convert_to_tensor(value=image)

    shape.check_static(image, has_rank=4)
    if size % 2 == 0:
      raise ValueError("The patch size is expected to be an odd value.")

    pixels = size**2
    channels = tf.shape(input=image)[-1]
    dtype = image.dtype
    patches = tf.image.extract_patches(
        image,
        sizes=(1, size, size, 1),
        strides=(1, 1, 1, 1),
        rates=(1, 1, 1, 1),
        padding="VALID")
    batches = tf.shape(input=patches)[:-1]
    new_shape = tf.concat((batches, (pixels, channels)), axis=-1)
    patches = tf.reshape(patches, shape=new_shape)
    mean = tf.reduce_mean(input_tensor=patches, axis=-2, keepdims=True)
    demean = patches - mean
    covariance = tf.matmul(demean, demean, transpose_a=True) / pixels
    regularizer = (eps / pixels) * tf.eye(channels, dtype=dtype)
    covariance_inv = tf.linalg.inv(covariance + regularizer)
    covariance_inv = asserts.assert_no_infs_or_nans(covariance_inv)
    mat = tf.matmul(tf.matmul(demean, covariance_inv), demean, transpose_b=True)
    return tf.eye(pixels, dtype=dtype) - (1.0 + mat) / pixels


class MattingTest(test_case.TestCase):

  @parameterized.parameters((3, 1), (3, 3), (5, 3), (5, 1))
  def test_build_matrices_jacobian_random(self, size, channels):
    """Tests the Jacobian of the build_matrices function."""
    tensor_shape = np.random.randint(size, 6, size=3)
    image_init = np.random.uniform(
        0.0, 1.0, size=tensor_shape.tolist() + [channels])

    with self.subTest(name="laplacian"):
      self.assert_jacobian_is_correct_fn(
          lambda image: matting.build_matrices(image, size=size)[0],
          [image_init])
    with self.subTest(name="pseudo_inverse"):
      self.assert_jacobian_is_correct_fn(
          lambda image: matting.build_matrices(image, size=size)[1],
          [image_init])

  @parameterized.parameters((3, 1), (3, 3), (5, 3), (5, 1))
  def test_build_matrices_laplacian_zero_rows_and_columns(self, size, channels):
    """Tests that the laplacian matrix rows and columns sum to zero."""
    tensor_shape = np.random.randint(size, 6, size=3)
    image_init = np.random.uniform(
        0.0, 1.0, size=tensor_shape.tolist() + [channels])
    image = tf.convert_to_tensor(value=image_init)

    laplacian, _ = matting.build_matrices(image, size=size)
    rows = tf.reduce_sum(input_tensor=laplacian, axis=-2)
    columns = tf.reduce_sum(input_tensor=laplacian, axis=-1)

    with self.subTest(name="rows"):
      self.assertAllClose(rows, tf.zeros_like(rows))
    with self.subTest(name="columns"):
      self.assertAllClose(columns, tf.zeros_like(columns))

  @parameterized.parameters((3, 1), (3, 3), (5, 3), (5, 1))
  def test_build_matrices_laplacian_versions(self, size, channels):
    """Compares two ways of computing the laplacian matrix."""
    tensor_shape = np.random.randint(size, 6, size=3)
    image_init = np.random.uniform(
        0.0, 1.0, size=tensor_shape.tolist() + [channels])
    image = tf.convert_to_tensor(value=image_init)

    laplacian_v1, _ = matting.build_matrices(image, size=size)
    laplacian_v2 = _laplacian_matrix(image, size=size)

    self.assertAllClose(laplacian_v1, laplacian_v2)

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
  def test_build_matrices_not_raised(self, size, *shapes):
    """Tests that the shape exceptions are not raised."""
    build_matrices = lambda image: matting.build_matrices(image, size=size)
    self.assert_exception_is_not_raised(build_matrices, shapes)

  @parameterized.parameters(
      ("tensor must have a rank of 4, but it has rank", 3, (1,)),
      ("tensor must have a rank of 4, but it has rank", 3, (1, 1, 1, 1, 1)),
      ("The patch size is expected to be an odd value.", 2, (1, 1, 1, 1)),
  )
  def test_build_matrices_raised(self, error_msg, size, *shapes):
    """Tests that the shape exceptions are properly raised."""
    build_matrices = lambda image: matting.build_matrices(image, size=size)
    self.assert_exception_is_raised(build_matrices, error_msg, shapes)

  @parameterized.parameters((3,), (5,))
  def test_linear_coefficients_jacobian_random(self, size):
    """Tests the Jacobian of the linear_coefficients function."""
    tensor_shape = np.random.randint(size, 6, size=3)
    matte_init = np.random.uniform(0.0, 1.0, size=tensor_shape.tolist() + [1])
    tensor_shape[1:3] -= (size - 1)
    num_coeffs = np.random.randint(2, 4)
    pseudo_inverse_init = np.random.uniform(
        0.0, 1.0, size=tensor_shape.tolist() + [num_coeffs, size**2])

    def a_fn(matte, pseudo_inverse):
      a, _ = matting.linear_coefficients(matte, pseudo_inverse)
      return a

    def b_fn(matte, pseudo_inverse):
      _, b = matting.linear_coefficients(matte, pseudo_inverse)
      return b

    with self.subTest(name="a"):
      self.assert_jacobian_is_correct_fn(a_fn,
                                         [matte_init, pseudo_inverse_init])
    with self.subTest(name="b"):
      self.assert_jacobian_is_correct_fn(b_fn,
                                         [matte_init, pseudo_inverse_init])

  @parameterized.parameters(
      ((None, None, None, 1), (None, None, None, 4, 9)),
      ((None, None, None, 1), (None, None, None, 2, 25)),
      ((1, 6, 6, 1), (1, 4, 4, 2, 9)),
      ((1, 10, 10, 1), (1, 6, 6, 2, 25)),
  )
  def test_linear_coefficients_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(matting.linear_coefficients, shapes)

  @parameterized.parameters(
      ("must have exactly 1 dimensions in axis -1", (1, 6, 6, 2),
       (1, 4, 4, 2, 9)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1),
       (2, 4, 4, 2, 9)),
  )
  def test_linear_coefficients_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(matting.linear_coefficients, error_msg,
                                    shapes)

  @parameterized.parameters((3,), (5,))
  def test_linear_coefficients_reconstruction_same_images(self, size):
    """Tests that the matte can be reconstructed by using the coefficients ."""
    tensor_shape = np.random.randint(size, 6, size=3).tolist()
    image = np.random.uniform(0.0, 1.0, size=tensor_shape + [1])

    _, pseudo_inverse = matting.build_matrices(image, size=size)
    a, b = matting.linear_coefficients(image, pseudo_inverse)
    reconstructed = matting.reconstruct(image, a, b)

    self.assertAllClose(image, reconstructed, atol=1e-4)

  @parameterized.parameters((3,), (5,))
  def test_linear_coefficients_reconstruction_opposite_images(self, size):
    """Tests that the matte can be reconstructed by using the coefficients ."""
    tensor_shape = np.random.randint(size, 6, size=3).tolist()
    image = np.random.uniform(0.0, 1.0, size=tensor_shape + [1])

    _, pseudo_inverse = matting.build_matrices(image, size=size)
    a, b = matting.linear_coefficients(1.0 - image, pseudo_inverse)
    reconstructed = matting.reconstruct(image, a, b)

    self.assertAllClose(1.0 - image, reconstructed, atol=1e-4)

  @parameterized.parameters((3,), (5,))
  def test_loss_jacobian_random(self, size):
    """Tests the Jacobian of the matting loss function."""
    tensor_shape = np.random.randint(size, 6, size=3)
    matte_init = np.random.uniform(0.0, 1.0, size=tensor_shape.tolist() + [1])
    tensor_shape[1:3] -= (size - 1)
    laplacian_init = np.random.uniform(
        0.0, 1.0, size=tensor_shape.tolist() + [size**2, size**2])

    with self.subTest(name="matte"):
      self.assert_jacobian_is_correct_fn(matting.loss,
                                         [matte_init, laplacian_init])

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

  @parameterized.parameters((3,), (5,))
  def test_loss_opposite_images(self, size):
    """Tests that passing opposite images results in a loss close to 0.0."""
    tensor_shape = np.random.randint(size, 6, size=3).tolist()
    image = np.random.uniform(0.0, 1.0, size=tensor_shape + [1])

    laplacian, _ = matting.build_matrices(image, size=size)
    loss = matting.loss(1.0 - image, laplacian)

    self.assertAllClose(loss, 0.0, atol=1e-4)

  @parameterized.parameters((3,), (5,))
  def test_loss_same_images(self, size):
    """Tests that passing same images results in a loss close to 0.0."""
    tensor_shape = np.random.randint(size, 6, size=3).tolist()
    image = np.random.uniform(0.0, 1.0, size=tensor_shape + [1])

    laplacian, _ = matting.build_matrices(image, size=size)
    loss = matting.loss(image, laplacian)

    self.assertAllClose(loss, 0.0, atol=1e-4)

  @parameterized.parameters((3,), (5,))
  def test_loss_positive(self, size):
    """Tests that the loss is always greater or equal to 0.0."""
    tensor_shape = np.random.randint(size, 6, size=3).tolist()
    image = tf.random.uniform(minval=0.0, maxval=1.0, shape=tensor_shape + [3])
    matte = tf.random.uniform(minval=0.0, maxval=1.0, shape=tensor_shape + [1])

    laplacian, _ = matting.build_matrices(image, size=size)
    loss = matting.loss(matte, laplacian)

    self.assertAllGreaterEqual(loss, 0.0)

  @parameterized.parameters((1,), (3,))
  def test_reconstruct_jacobian_random(self, channels):
    """Tests the Jacobian of the reconstruct function."""
    tensor_shape = np.random.randint(1, 5, size=3).tolist()
    image_init = np.random.uniform(0.0, 1.0, size=tensor_shape + [channels])
    mul_init = np.random.uniform(0.0, 1.0, size=tensor_shape + [channels])
    add_init = np.random.uniform(0.0, 1.0, size=tensor_shape + [1])

    self.assert_jacobian_is_correct_fn(matting.reconstruct,
                                       [image_init, mul_init, add_init])

  @parameterized.parameters(
      ((None, None, None, 3), (None, None, None, 3), (None, None, None, 1)),
      ((1, 6, 6, 3), (1, 6, 6, 3), (1, 6, 6, 1)),
  )
  def test_reconstruct_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(matting.reconstruct, shapes)

  @parameterized.parameters(
      ("tensor must have a rank of 4, but it has rank", (1, 6, 6), (1, 6, 6, 2),
       (1, 6, 6, 1)),
      ("tensor must have a rank of 4, but it has rank", (1, 6, 6, 2), (1, 6, 6),
       (1, 6, 6, 1)),
      ("tensor must have a rank of 4, but it has rank", (1, 6, 6, 2),
       (1, 6, 6, 2), (1, 6, 6)),
      ("must have exactly 1 dimensions in axis -1", (1, 6, 6, 2), (1, 6, 6, 2),
       (1, 6, 6, 2)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (1, 6, 6, 4),
       (1, 6, 6, 1)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (1, 4, 6, 1),
       (1, 6, 6, 1)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (1, 6, 6, 1),
       (1, 4, 6, 1)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (1, 6, 4, 1),
       (1, 6, 6, 1)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (1, 6, 6, 1),
       (1, 6, 4, 1)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (4, 6, 6, 1),
       (1, 6, 6, 1)),
      ("Not all batch dimensions are identical.", (1, 6, 6, 1), (1, 6, 6, 1),
       (4, 6, 6, 1)),
  )
  def test_reconstruct_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(matting.reconstruct, error_msg, shapes)


if __name__ == "__main__":
  test_case.main()
