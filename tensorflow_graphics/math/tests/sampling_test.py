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
"""Tests for feature representations."""
from absl.testing import parameterized
from tensorflow_graphics.math import sampling
from tensorflow_graphics.util import test_case


class SamplingTest(test_case.TestCase):

  @parameterized.parameters(
      (128, (), ()),
      (128, (2), (2)),
      (128, (2, 4), (2, 4)),
  )
  def test_regular_1d_shape_exception_not_raised(self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(sampling.regular_1d,
                                        shape,
                                        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 3), (3, 2)),
  )
  def test_regular_1d_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.regular_1d,
                                    error_msg,
                                    shapes=shape,
                                    num_samples=128)

  @parameterized.parameters(
      (128, (), ()),
      (128, (2), (2)),
      (128, (2, 4), (2, 4)),
  )
  def test_regular_inverse_1d_shape_exception_not_raised(
      self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(sampling.regular_inverse_1d,
                                        shape,
                                        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 3), (3, 2)),
  )
  def test_regular_inverse_1d_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.regular_inverse_1d,
                                    error_msg,
                                    shapes=shape,
                                    num_samples=128)

  @parameterized.parameters(
      (128, (), ()),
      (128, (2), (2)),
      (128, (2, 4), (2, 4)),
  )
  def test_uniform_1d_shape_exception_not_raised(self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(sampling.uniform_1d,
                                        shape,
                                        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 3), (3, 2)),
  )
  def test_uniform_1d_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.uniform_1d,
                                    error_msg,
                                    shapes=shape,
                                    num_samples=128)

  @parameterized.parameters(
      (128, (), ()),
      (128, (2), (2)),
      (128, (2, 4), (2, 4)),
  )
  def test_stratified_1d_shape_exception_not_raised(self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(sampling.stratified_1d,
                                        shape,
                                        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 3), (3, 2)),
  )
  def test_stratified_1d_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.stratified_1d,
                                    error_msg,
                                    shapes=shape,
                                    num_samples=128)

  @parameterized.parameters(
      (128, (), ()),
      (128, (2), (2)),
      (128, (2, 4), (2, 4)),
  )
  def test_stratified_geomspace_1d_shape_exception_not_raised(
      self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(sampling.stratified_geomspace_1d,
                                        shape,
                                        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 3), (3, 2)),
  )
  def test_stratified_geomspace_1d_shape_exception_raised(
      self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.stratified_geomspace_1d,
                                    error_msg,
                                    shapes=shape,
                                    num_samples=128)

  @parameterized.parameters(
      (128, (12), (12)),
      (128, (2, 12), (2, 12)),
  )
  def test_inverse_transform_sampling_1d_shape_exception_not_raised(
      self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(sampling.inverse_transform_sampling_1d,
                                        shape,
                                        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 12), (2, 12)),
      ("must have the same number of dimensions", (3, 12), (3, 11)),
  )
  def test_inverse_transform_sampling_1d_shape_exception_raised(self,
                                                                error_msg,
                                                                *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.inverse_transform_sampling_1d,
                                    error_msg,
                                    shape,
                                    num_samples=128)

  @parameterized.parameters(
      (128, (12), (12), (12)),
      (128, (2, 12), (2, 12), (2, 12)),
  )
  def test_inverse_transform_stratified_1d_shape_exception_not_raised(
      self, num_samples, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        sampling.inverse_transform_stratified_1d, shape,
        num_samples=num_samples)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.",
       (3, 12), (2, 12), (3, 12)),
      ("must have the same number of dimensions", (3, 12), (3, 11), (3, 12)),
  )
  def test_inverse_transform_stratified_1d_shape_exception_raised(self,
                                                                  error_msg,
                                                                  *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(sampling.inverse_transform_stratified_1d,
                                    error_msg,
                                    shape,
                                    num_samples=128)

if __name__ == "__main__":
  test_case.main()
