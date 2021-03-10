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
r"""This module implements a Levenberg-Marquardt optimizer.

Minimizes \\(\min_{\mathbf{x}} \sum_i \|\mathbf{r}_i(\mathbf{x})\|^2_2\\) where
\\(\mathbf{r}_i(\mathbf{x})\\)
are the residuals. This function implements Levenberg-Marquardt, an iterative
process that linearizes the residuals and iteratively finds a displacement
\\(\Delta \mathbf{x}\\) such that at iteration \\(t\\) an update
\\(\mathbf{x}_{t+1} = \mathbf{x}_{t} + \Delta \mathbf{x}\\) improving the
loss can be computed. The displacement is computed by solving an optimization
problem
\\(\min_{\Delta \mathbf{x}} \sum_i
\|\mathbf{J}_i(\mathbf{x}_{t})\Delta\mathbf{x} +
\mathbf{r}_i(\mathbf{x}_t)\|^2_2 + \lambda\|\Delta \mathbf{x} \|_2^2\\) where
\\(\mathbf{J}_i(\mathbf{x}_{t})\\) is the Jacobian of \\(\mathbf{r}_i\\)
computed at \\(\mathbf{x}_t\\), and \\(\lambda\\) is a scalar weight.

More details on Levenberg-Marquardt can be found on [this page.]
(https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from tensorflow_graphics.util import export_api


def _values_and_jacobian(residuals, variables):
  """Computes the residual values and the Jacobian matrix.

  Args:
    residuals: A list of residuals.
    variables: A list of variables.

  Returns:
    The residual values and the Jacobian matrix.
  """

  def _compute_residual_values(residuals, variables):
    """Computes the residual values."""
    return tf.concat([
        tf.reshape(residual(*variables), shape=(-1,)) for residual in residuals
    ],
                     axis=-1)

  def _compute_jacobian(values, variables, tape):
    """Computes the Jacobian matrix."""
    jacobians = tape.jacobian(
        values, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return tf.concat([
        tf.reshape(jacobian, shape=(tf.shape(input=jacobian)[0], -1))
        for jacobian in jacobians
    ],
                     axis=-1)

  with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
    for variable in variables:
      tape.watch(variable)
    values = _compute_residual_values(residuals, variables)
  jacobian = _compute_jacobian(values, variables, tape)
  del tape
  values = tf.expand_dims(values, axis=-1)
  return values, jacobian


def minimize(residuals,
             variables,
             max_iterations,
             regularizer=1e-20,
             regularizer_multiplier=10.0,
             callback=None,
             name="levenberg_marquardt_minimize"):
  r"""Minimizes a set of residuals in the least-squares sense.

  Args:
    residuals: A residual or a list/tuple of residuals. A residual is a Python
      `callable`.
    variables: A variable or a list or tuple of variables defining the starting
      point of the minimization.
    max_iterations: The maximum number of iterations.
    regularizer: The regularizer is used to damped the stepsize when the
      iterations are becoming unstable. The bigger the regularizer is the
      smaller the stepsize becomes.
    regularizer_multiplier: If an iteration does not decrease the objective a
      new regularizer is computed by scaling it by this multiplier.
    callback: A callback function that will be called at each iteration. In
      graph mode the callback should return an op or list of ops that will
      execute the callback logic. The callback needs to be of the form
      f(iteration, objective_value, variables). A callback is a Python
      `callable`. The callback could be used for logging, for example if one
      wants to print the objective value at each iteration.
    name: A name for this op. Defaults to "levenberg_marquardt_minimize".

  Returns:
    The value of the objective function and variables attained at the final
    iteration of the minimization procedure.

  Raises:
    ValueError: If max_iterations is not at least 1.
    InvalidArgumentError: This exception is only raised in graph mode if the
    Cholesky decomposition is not successful. One likely fix is to increase
    the regularizer. In eager mode this exception is catched and the regularizer
    is increased automatically.

  Examples:

    ```python
    x = tf.constant(np.random.random_sample(size=(1,2)), dtype=tf.float32)
    y = tf.constant(np.random.random_sample(size=(3,1)), dtype=tf.float32)

    def f1(x, y):
      return x + y

    def f2(x, y):
      return x * y

    def callback(iteration, objective_value, variables):
      def print_output(iteration, objective_value, *variables):
        print("Iteration:", iteration, "Objective Value:", objective_value)
        for variable in variables:
          print(variable)
      inp = [iteration, objective_value] + variables
      return tf.py_function(print_output, inp, [])

    minimize_op = minimize(residuals=(f1, f2),
                           variables=(x, y),
                           max_iterations=10,
                           callback=callback)

    if not tf.executing_eagerly():
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(minimize_op)
    ```
  """
  if not isinstance(variables, (tuple, list)):
    variables = [variables]
  with tf.name_scope(name):
    if not isinstance(residuals, (tuple, list)):
      residuals = [residuals]
    if isinstance(residuals, tuple):
      residuals = list(residuals)
    if isinstance(variables, tuple):
      variables = list(variables)
    variables = [tf.convert_to_tensor(value=variable) for variable in variables]
    multiplier = tf.constant(regularizer_multiplier, dtype=variables[0].dtype)

    if max_iterations <= 0:
      raise ValueError("'max_iterations' needs to be at least 1.")

    def _cond(iteration, regularizer, objective_value, variables):
      """Returns whether any iteration still needs to be performed."""
      del regularizer, objective_value, variables
      return iteration < max_iterations

    def _body(iteration, regularizer, objective_value, variables):
      """Main optimization loop."""
      iteration += tf.constant(1, dtype=tf.int32)
      values, jacobian = _values_and_jacobian(residuals, variables)
      # Solves the normal equation.
      try:
        updates = tf.linalg.lstsq(jacobian, values, l2_regularizer=regularizer)
        shapes = [tf.shape(input=variable) for variable in variables]
        splits = [tf.reduce_prod(input_tensor=shape) for shape in shapes]
        updates = tf.split(tf.squeeze(updates, axis=-1), splits)
        new_variables = [
            variable - tf.reshape(update, shape)
            for variable, update, shape in zip(variables, updates, shapes)
        ]
        new_objective_value = tf.reduce_sum(input_tensor=[
            tf.nn.l2_loss(residual(*new_variables)) for residual in residuals
        ])
        # If the new estimated solution does not decrease the objective value,
        # no updates are performed, but a new regularizer is computed.
        cond = tf.less(new_objective_value, objective_value)
        regularizer = tf.where(cond, x=regularizer, y=regularizer * multiplier)
        objective_value = tf.where(
            cond, x=new_objective_value, y=objective_value)
        variables = [
            tf.where(cond, x=new_variable, y=variable)
            for variable, new_variable in zip(variables, new_variables)
        ]
      # Note that catching InvalidArgumentError will only work in eager mode.
      except tf.errors.InvalidArgumentError:
        regularizer *= multiplier
      if callback is not None:
        callback_ops = callback(iteration, objective_value, variables)
        if callback_ops is not None:
          if not isinstance(callback_ops, (tuple, list)):
            callback_ops = [callback_ops]
          with tf.control_dependencies(callback_ops):
            iteration = tf.identity(iteration)
            objective_value = tf.identity(objective_value)
            variables = [tf.identity(v) for v in variables]
      return iteration, regularizer, objective_value, variables

    starting_value = tf.reduce_sum(input_tensor=[
        tf.nn.l2_loss(residual(*variables)) for residual in residuals
    ])
    dtype = variables[0].dtype
    initial = (
        tf.constant(0, dtype=tf.int32),  # Initial iteration number.
        tf.constant(regularizer, dtype=dtype),  # Initial regularizer.
        starting_value,  # Initial objective value.
        variables,  # Initial variables.
    )
    _, _, final_objective_value, final_variables = tf.while_loop(
        cond=_cond, body=_body, loop_vars=initial, parallel_iterations=1)
    return final_objective_value, final_variables


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
