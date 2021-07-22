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
"""Implements an ExponentialMovingAverage class that is checkpointable."""

from typing import Sequence

import tensorflow as tf


class ExponentialMovingAverage(tf.Module):
  """Exponential moving average.

  This class is a checkpointable implementation of a subset of the functionality
  provided by tf.train.ExponentialMovingAverage. The tf version is not
  checkpointable due to use of tf.Variable.ref() to associate tf.Variables
  objects to their corresponding averages
  (cf. https://github.com/tensorflow/tensorflow/issues/38452). This version uses
  the order of the tf.Variable objects in a sequence to associate the variables
  with their averages.

  Note: This class offers less functionality than the tensorflow version and it
  is only implemented for replica context.

  Attributes:
    averaged_variables: A sequence of tf.Variables that stores the averages for
      the variables. They are associated to the new values that are provided to
      ExponentialMovingAverage.apply() by the order in the sequence. If None a
      call to ExponentialMovingAverage.apply() initializes the variable before
      applying the update.
  """

  def __init__(self, decay: float = 0.999):
    """Initializes exponential moving average.

    Args:
      decay: The decay rate of the exponential moving average.
    """
    self.averaged_variables: Sequence[tf.Variable] = None
    self._decay = decay

  def _ema_assign_fn(self, variable: tf.Variable, value: tf.Tensor):
    """Updates the exponential moving average for a single variable."""
    return variable.assign(self._decay * variable + (1.0 - self._decay) * value)

  def _apply_values(self, variables: Sequence[tf.Variable]):
    """Applies the new values to the exponential moving averages."""

    def merge_fn(strategy: tf.distribute.Strategy, variable: tf.Variable,
                 value: tf.Tensor):
      value = strategy.extended.reduce_to(tf.distribute.ReduceOp.MEAN, value,
                                          variable)
      strategy.extended.update(variable, self._ema_assign_fn, args=(value,))

    replica_context = tf.distribute.get_replica_context()

    if replica_context:
      for variable_ema, variable in zip(self.averaged_variables, variables):
        replica_context.merge_call(merge_fn, args=(variable_ema, variable))
    else:
      raise NotImplementedError(
          'Cross-replica context version not implemented.')

  def apply(self, variables: Sequence[tf.Variable]):
    """Applies new values to the averages.

    This function is called to update the averages with new values. If the
    variables for the averages have not been created before this function
    creates new variables for the averages before the update.

    Args:
      variables: The variables storing the values to apply to the averages. The
        sequence is assumed to have the same order of the variables as the
        averages stored in self.averaged_variables. If self.averaged_variables
        is None it gets initialized with a new sequence of variables with the
        values of the provided variables as initial value.
    """
    if self.averaged_variables is None:
      with tf.init_scope():
        strategy = tf.distribute.get_strategy()
        self.averaged_variables = []

        for variable in variables:
          with strategy.extended.colocate_vars_with(variable):
            self.averaged_variables.append(
                tf.Variable(initial_value=variable.read_value()))
    self._apply_values(variables)
