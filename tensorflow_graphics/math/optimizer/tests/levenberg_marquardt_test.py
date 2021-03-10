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
"""Tests for the Levenberg Marquardt optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from six.moves import zip
import tensorflow as tf

from tensorflow_graphics.math.optimizer import levenberg_marquardt
from tensorflow_graphics.util import test_case


class OptimizerTest(test_case.TestCase):

  @parameterized.parameters(
      (4, (1,)),
      (3, (1, 1)),
      (2, (2,)),
      (1, (2, 2)),
  )
  def test_minimize_not_raised(self, max_iterations, *shapes):
    """Tests that the shape exceptions are not raised."""

    def callback(iteration, objective_value, variables):
      del iteration, objective_value, variables

    def optimize(*variables):
      levenberg_marquardt.minimize(
          lambda x: x, variables, max_iterations, callback=callback)

    self.assert_exception_is_not_raised(optimize, shapes)

  @parameterized.parameters(
      ("'max_iterations' needs to be at least 1.", -1, (1,)),
      ("'max_iterations' needs to be at least 1.", 0, (1,)),
  )
  def test_minimize_raised(self, error_msg, max_iterations, *shape):
    """Tests that the exception is raised."""

    def optimize(*variables):
      levenberg_marquardt.minimize(lambda x: x, variables, max_iterations)

    self.assert_exception_is_raised(optimize, error_msg, shape)

  @parameterized.parameters(
      ([lambda x: x], [[1.]], 1, 0., [[0.]]),
      ((lambda x: x,), ([1.],), 1, 0., [[0.]]),
      ([lambda x, y: x, lambda x, y: y], [[1.], [1.]], 1, 0., [[0.], [0.]]),
      ((lambda x, y: x, lambda x, y: y), ([1.], [1.]), 1, 0., [[0.], [0.]]),
  )
  def test_minimize_preset(self, residuals, variables, max_iterations,
                           final_objective_value, final_variables):
    """Tests the output of the minimization for some presets."""
    objective_value, output_variables = levenberg_marquardt.minimize(
        residuals, variables, max_iterations)

    with self.subTest(name="objective_value"):
      self.assertAllClose(objective_value, final_objective_value)

    with self.subTest(name="variables"):
      for output_variable, final_variable in zip(output_variables,
                                                 final_variables):
        self.assertAllClose(output_variable, final_variable)

  def test_minimize_callback(self):
    """Callback should run at the end of every iteration."""
    variables = np.random.uniform(low=-1.0, high=1.0, size=[1])
    saved_objective = [None]
    saved_iteration = [None]
    iteration_count = [0]

    def callback(iteration, objective_value, variables):
      del variables

      def save(iteration, objective_value):
        saved_objective[0] = objective_value
        saved_iteration[0] = iteration
        iteration_count[0] += 1

      return tf.py_function(save, [iteration, objective_value], [])

    final_objective, _ = levenberg_marquardt.minimize(
        lambda x: x, variables, max_iterations=5, callback=callback)
    final_saved_objective = tf.py_function(lambda: saved_objective[0], [],
                                           tf.float64)
    final_saved_iteration = tf.py_function(lambda: saved_iteration[0], [],
                                           tf.int32)
    final_iteration_count = tf.py_function(lambda: iteration_count[0], [],
                                           tf.int32)

    with self.subTest(name="objective_value"):
      self.assertAllClose(final_objective, final_saved_objective)

    with self.subTest(name="iterations"):
      self.assertAllEqual(final_saved_iteration, final_iteration_count)

  def test_minimize_ill_conditioned_not_raised(self):
    """Optimizing an ill conditioned problem should not raise an exception."""
    if not tf.executing_eagerly():
      return

    def f1(x, y):
      return x * y * 10000.0

    def f2(x, y):
      return x * y * 0.0001

    x = (1.,)
    y = (1.,)
    try:
      self.evaluate(
          levenberg_marquardt.minimize(
              residuals=(f1, f2),
              variables=(x, y),
              max_iterations=1,
              regularizer=1e-20))
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def test_minimize_linear_residuals_random(self):
    """Optimizing linear residuals should give the minimum in 1 step."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    variables = np.random.uniform(low=-1.0, high=1.0, size=tensor_shape)
    objective_value, variables = levenberg_marquardt.minimize(
        lambda x: x, variables, max_iterations=1)

    with self.subTest(name="objective_value"):
      self.assertAllClose(objective_value, tf.zeros_like(objective_value))

    with self.subTest(name="variables"):
      for variable in variables:
        self.assertAllClose(variable, tf.zeros_like(variable))


if __name__ == "__main__":
  test_case.main()
