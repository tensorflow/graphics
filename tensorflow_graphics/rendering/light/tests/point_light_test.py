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
"""Tests for point light."""

import math

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.light import point_light
from tensorflow_graphics.util import test_case


def fake_brdf(incoming_light_direction, outgoing_light_direction,
              surface_point_normal):
  del incoming_light_direction, surface_point_normal  # Unused.
  return outgoing_light_direction


def returning_zeros_brdf(incoming_light_direction, outgoing_light_direction,
                         surface_point_normal):
  del incoming_light_direction, outgoing_light_direction  # Unused.
  return tf.zeros_like(surface_point_normal)


def random_tensor(tensor_shape):
  return np.random.uniform(low=-100.0, high=100.0, size=tensor_shape)


class PointLightTest(test_case.TestCase):

  @parameterized.parameters(
      # Light direction is parallel to the surface normal.
      ([1.], [[0., 0., 1.]], [2., 0., 0.], [1.0 / (4. * math.pi), 0., 0.]),
      # Light direction is parallel to the surface normal and the reflected
      # light fall off is included in the calculation.
      ([1.], [[0., 0., 1.]], [2., 0., 0.], \
       [0.25 / (4. * math.pi), 0., 0.], True),
      # Light direction is perpendicular to the surface normal.
      ([1.], [[3., 0., 0.]], [1., 2., 3.], [0., 0., 0.]),
      # Angle between surface normal and the incoming light direction is pi/3.
      ([1.], [[math.sqrt(3), 0., 1.]], \
       [0., 1., 0.], [0., 0.125 / (4. * math.pi), 0.]),
      # Angle between surface normal and the incoming light direction is pi/4.
      ([1.], [[0., 1., 1.]], [1., 1., 0.],
       [0.25 / (4. * math.pi), 0.25 / (4. * math.pi), 0.]),
      # Light has 3 radiances.
      ([2., 4., 1.], [[0., 1., 1.]], [1., 1., 0.],
       [0.5 / (4. * math.pi), 1. / (4. * math.pi), 0.]),
      # Light is behind the surface.
      ([1.], [[0., 0., -2.]], [7., 0., 0.], [0., 0., 0.]),
      # Observation point is behind the surface.
      ([1.], [[0., 0., 2.]], [5., 0., -2.], [0., 0., 0.]),
      # Light and observation point are behind the surface.
      ([1.], [[0., 0., -2.]], [5., 0., -2.], [0., 0., 0.]),
  )
  def test_estimate_radiance_preset(self,
                                    light_radiance,
                                    light_pos,
                                    observation_pos,
                                    expected_result,
                                    reflected_light_fall_off=False):
    """Tests the output of estimate radiance function with various parameters.

    In this test the point on the surface is always [0, 0, 0] ,the surface
    normal is [0, 0, 1] and the fake brdf function returns the (normalized)
    direction of the outgoing light as its output.

    Args:
     light_radiance: An array of size K representing the point light radiances.
     light_pos: An array of size [3,] representing the point light positions.
     observation_pos: An array of size [3,] representing the observation point.
     expected_result: An array of size [3,] representing the expected result of
       the estimated reflected radiance function.
     reflected_light_fall_off: A boolean specifying whether or not to include
       the fall off of the reflected light in the calculation. Defaults to
       False.
    """
    tensor_size = np.random.randint(1, 3) + 1
    tensor_shape = np.random.randint(1, 10, size=tensor_size).tolist()
    lights_tensor_size = np.random.randint(1, 3) + 1
    lights_tensor_shape = np.random.randint(
        1, 10, size=lights_tensor_size).tolist()
    point_light_radiance = np.tile(light_radiance, lights_tensor_shape + [1])
    point_light_position = np.tile(light_pos, lights_tensor_shape + [1])
    surface_point_normal = np.tile([0.0, 0.0, 1.0], tensor_shape + [1])
    surface_point_position = np.tile([0.0, 0.0, 0.0], tensor_shape + [1])
    observation_point = np.tile(observation_pos, tensor_shape + [1])
    expected = np.tile(expected_result,
                       tensor_shape + lights_tensor_shape + [1])

    pred = point_light.estimate_radiance(
        point_light_radiance,
        point_light_position,
        surface_point_position,
        surface_point_normal,
        observation_point,
        fake_brdf,
        reflected_light_fall_off=reflected_light_fall_off)

    self.assertAllClose(expected, pred)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_estimate_radiance_jacobian_random(self):
    """Tests the Jacobian of the point lighting equation."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 10, size=tensor_size).tolist()
    light_tensor_size = np.random.randint(1, 3)
    lights_tensor_shape = np.random.randint(
        1, 10, size=light_tensor_size).tolist()
    point_light_radiance_init = random_tensor(lights_tensor_shape + [1])
    point_light_position_init = random_tensor(lights_tensor_shape + [3])
    surface_point_position_init = random_tensor(tensor_shape + [3])
    surface_point_normal_init = random_tensor(tensor_shape + [3])
    observation_point_init = random_tensor(tensor_shape + [3])

    def estimate_radiance_fn(point_light_position, surface_point_position,
                             surface_point_normal, observation_point):
      return point_light.estimate_radiance(point_light_radiance_init,
                                           point_light_position,
                                           surface_point_position,
                                           surface_point_normal,
                                           observation_point, fake_brdf)

    self.assert_jacobian_is_correct_fn(estimate_radiance_fn, [
        point_light_position_init, surface_point_position_init,
        surface_point_normal_init, observation_point_init
    ])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_estimate_radiance_jacobian_preset(self):
    """Tests the Jacobian of the point lighting equation.

    Verifies that the Jacobian of the point lighting equation is correct when
    the light direction is orthogonal to the surface normal.
    """
    delta = 1e-5
    point_light_radiance_init = np.array(1.0).reshape((1, 1))
    point_light_position_init = np.array((delta, 1.0, 0.0)).reshape((1, 3))
    surface_point_position_init = np.array((0.0, 0.0, 0.0))
    surface_point_normal_init = np.array((1.0, 0.0, 0.0))
    observation_point_init = np.array((delta, 3.0, 0.0))

    def estimate_radiance_fn(point_light_position, surface_point_position,
                             surface_point_normal, observation_point):
      return point_light.estimate_radiance(point_light_radiance_init,
                                           point_light_position,
                                           surface_point_position,
                                           surface_point_normal,
                                           observation_point, fake_brdf)

    self.assert_jacobian_is_correct_fn(estimate_radiance_fn, [
        point_light_position_init, surface_point_position_init,
        surface_point_normal_init, observation_point_init
    ])

  @parameterized.parameters(
      ((1, 1), (1, 3), (3,), (3,), (3,)),
      ((4, 1, 1), (4, 1, 3), (1, 3), (1, 3), (1, 3)),
      ((3, 2, 1), (3, 2, 3), (2, 3), (2, 3), (2, 3)),
      ((1, 1), (3,), (1, 3), (1, 2, 3), (1, 3)),
      ((4, 5, 1), (3, 4, 5, 3), (1, 3), (1, 2, 2, 3), (1, 2, 3)),
      ((1,), (1, 2, 2, 3), (1, 2, 3), (1, 3), (3,)),
  )
  def test_estimate_radiance_shape_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        point_light.estimate_radiance, shape, brdf=returning_zeros_brdf)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 1), (3,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (5, 1), (5, 2), (3,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 4), (3,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (1,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (2,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (4,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (3,), (1,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (3,), (2,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (3,), (4,),
       (3,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (3,), (3,),
       (4,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (3,), (3,),
       (2,)),
      ("must have exactly 3 dimensions in axis -1", (1, 1), (1, 3), (3,), (3,),
       (1,)),
      ("Not all batch dimensions are broadcast-compatible.", (1, 3, 1),
       (1, 3, 3), (2, 3), (4, 3), (3,)),
      ("Not all batch dimensions are broadcast-compatible.", (1, 3, 1),
       (1, 4, 3), (2, 3), (3,), (3,)),
  )
  def test_estimate_radiance_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(
        point_light.estimate_radiance,
        error_msg,
        shape,
        brdf=returning_zeros_brdf)

  def test_estimate_radiance_value_exceptions_raised(self):
    """Tests that the value exceptions are raised correctly."""
    point_light_radiance = random_tensor(tensor_shape=(1, 1))
    point_light_position = random_tensor(tensor_shape=(1, 3))
    surface_point_position = random_tensor(tensor_shape=(3,))
    surface_point_normal = random_tensor(tensor_shape=(3,))
    observation_point = random_tensor(tensor_shape=(3,))

    # Verify that an InvalidArgumentError is raised as the given
    # surface_point_normal is not normalized.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          point_light.estimate_radiance(point_light_radiance,
                                        point_light_position,
                                        surface_point_position,
                                        surface_point_normal, observation_point,
                                        returning_zeros_brdf))


if __name__ == "__main__":
  test_case.main()
