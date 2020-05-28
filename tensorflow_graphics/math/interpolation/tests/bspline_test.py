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
"""Tests for bspline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.math.interpolation import bspline
from tensorflow_graphics.util import test_case


class BSplineTest(test_case.TestCase):

  @parameterized.parameters((0.0, (1.0,)), (1.0, (1.0,)))
  def test_constant_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 0 return expected values."""
    self.assertAllClose(bspline._constant(position), weights)  # pylint: disable=protected-access

  @parameterized.parameters((0.0, (1.0, 0.0)), (1.0, (0.0, 1.0)))
  def test_linear_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 1 return expected values."""
    self.assertAllClose(bspline._linear(position), weights)  # pylint: disable=protected-access

  @parameterized.parameters((0.0, (0.5, 0.5, 0.0)), (1.0, (0.0, 0.5, 0.5)))
  def test_quadratic_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 2 return expected values."""
    self.assertAllClose(bspline._quadratic(position), weights)  # pylint: disable=protected-access

  @parameterized.parameters((0.0, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0, 0.0)),
                            (1.0, (0.0, 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)))
  def test_cubic_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 3 return expected values."""
    self.assertAllClose(bspline._cubic(position), weights)  # pylint: disable=protected-access

  @parameterized.parameters(
      (0.0, (1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0, 0.0)),
      (1.0, (0.0, 1.0 / 24.0, 11.0 / 24.0, 11.0 / 24.0, 1.0 / 24.0)))
  def test_quartic_basis_boundary_values(self, position, weights):
    """Tests that basis functions of degree 4 return expected values."""
    self.assertAllClose(bspline._quartic(position), weights)  # pylint: disable=protected-access

  @parameterized.parameters(
      (((0.5,), (1.5,), (2.5,)), (((0.5, 0.5),), ((0.5, 0.5),), ((0.5, 0.5),)),
       (((0,), (1,), (2,))), 1, True),
      ((0.0, 1.0), ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5)), (0, 0), 2, False),
  )
  def test_knot_weights_sparse_mode_preset(self, positions, gt_weights,
                                           gt_shifts, degree, cyclical):
    """Tests that sparse mode returns correct results."""
    weights, shifts = bspline.knot_weights(
        positions,
        num_knots=3,
        degree=degree,
        cyclical=cyclical,
        sparse_mode=True)

    self.assertAllClose(weights, gt_weights)
    self.assertAllClose(shifts, gt_shifts)

  @parameterized.parameters(
      (((0.5,),), (((0.5, 0.5, 0.0),),), 1),
      (((1.5,),), (((0.0, 0.5, 0.5),),), 1),
      (((2.5,),), (((0.5, 0.0, 0.5),),), 1),
      (((0.5,), (1.5,), (2.5,)),
       (((1.0 / 8.0, 0.75, 1.0 / 8.0),), ((1.0 / 8.0, 1.0 / 8.0, 0.75),),
        ((0.75, 1.0 / 8.0, 1.0 / 8.0),)), 2),
  )
  def test_knot_weights_preset(self, position, weights, degree):
    """Tests that knot weights are correct when degree < num_knots - 1."""
    self.assertAllClose(
        bspline.knot_weights(
            position, num_knots=3, degree=degree, cyclical=True), weights)

  @parameterized.parameters((((0.0,), (0.25,), (0.5,), (0.75,)),))
  def test_full_degree_non_cyclical_knot_weights(self, positions):
    """Tests that noncyclical weights are correct when using max degree."""
    cyclical_weights = bspline.knot_weights(
        positions=positions, num_knots=3, degree=2, cyclical=True)
    noncyclical_weights = bspline.knot_weights(
        positions=positions, num_knots=3, degree=2, cyclical=False)

    self.assertAllClose(cyclical_weights, noncyclical_weights)

  @parameterized.parameters(
      ("must have the same number of dimensions", ((None, 2), (None, 3, 3))),
      ("must have the same number of dimensions", ((2,), (3,))),
  )
  def test_interpolate_with_weights_exception_is_raised(self, error_msg,
                                                        shapes):
    """Tests that exception is raised when wrong number of knots is given."""
    self.assert_exception_is_raised(
        bspline.interpolate_with_weights, error_msg, shapes=shapes)

  @parameterized.parameters(
      (((0.5,), (0.0,), (0.9,)), (((0.5, 1.5), (1.5, 1.5), (2.5, 3.5)),)))
  def test_interpolate_with_weights_preset(self, positions, knots):
    """Tests that interpolate_with_weights works correctly."""
    degree = 1
    cyclical = False
    interp1 = bspline.interpolate(knots, positions, degree, cyclical)
    weights = bspline.knot_weights(positions, 2, degree, cyclical)
    interp2 = bspline.interpolate_with_weights(knots, weights)

    self.assertAllClose(interp1, interp2)

  @parameterized.parameters(
      (1, 2),
      (1, None),
      (2, 2),
      (2, None),
      (3, 2),
      (3, None),
      (4, 2),
      (4, None),
  )
  def test_knot_weights_exception_is_not_raised(self, positions_rank, dims):
    shapes = ([dims] * positions_rank,)

    self.assert_exception_is_not_raised(
        bspline.knot_weights,
        shapes=shapes,
        num_knots=3,
        degree=2,
        cyclical=True)

  @parameterized.parameters(
      ("Degree should be between 0 and 4.", 6, -1),
      ("Degree should be between 0 and 4.", 6, 5),
      ("Degree cannot be >= number of knots.", 2, 2),
      ("Degree cannot be >= number of knots.", 2, 3),
  )
  def test_knot_weights_exception_is_raised(self, error_msg, num_knots, degree):
    self.assert_exception_is_raised(
        bspline.knot_weights,
        error_msg,
        shapes=((10, 1),),
        num_knots=num_knots,
        degree=degree,
        cyclical=True)

  @parameterized.parameters(
      (1, 0, True),
      (1, 0, False),
      (2, 1, True),
      (2, 1, False),
      (3, 1, True),
      (3, 1, False),
      (3, 2, True),
      (3, 2, False),
      (4, 1, True),
      (4, 1, False),
      (4, 3, True),
      (4, 3, False),
      (5, 1, True),
      (5, 1, False),
      (5, 4, True),
      (5, 4, False),
  )
  def test_knot_weights_jacobian_is_correct(self, num_knots, degree, cyclical):
    """Tests that Jacobian is correct."""
    positions_init = np.random.random_sample((10, 1))
    scale = num_knots if cyclical else num_knots - degree
    positions_init *= scale

    def dense_mode_fn(positions):
      return bspline.knot_weights(
          positions=positions,
          num_knots=num_knots,
          degree=degree,
          cyclical=cyclical,
          sparse_mode=False)

    def sparse_mode_fn(positions):
      return bspline.knot_weights(
          positions=positions,
          num_knots=num_knots,
          degree=degree,
          cyclical=cyclical,
          sparse_mode=True)[0]

    with self.subTest(name="dense_mode"):
      self.assert_jacobian_is_correct_fn(dense_mode_fn, [positions_init])

    with self.subTest(name="sparse_mode"):
      self.assert_jacobian_is_correct_fn(sparse_mode_fn, [positions_init])


if __name__ == "__main__":
  test_case.main()
