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
"""Tests for gan.exponential_moving_average."""

import tensorflow as tf

from tensorflow_graphics.projects.gan import exponential_moving_average


class ExponentialMovingAverageTest(tf.test.TestCase):

  def test_decay_one_values_are_from_initialization(self):
    ema = exponential_moving_average.ExponentialMovingAverage(decay=1.0)
    initial_value = 2.0
    variable = tf.Variable(initial_value)

    ema.apply((variable,))
    variable.assign(3.0)
    ema.apply((variable,))

    self.assertAllClose(ema.averaged_variables[0], initial_value)

  def test_decay_zero_returns_last_value(self):
    ema = exponential_moving_average.ExponentialMovingAverage(decay=0.0)
    final_value = 3.0
    variable = tf.Variable(2.0)

    ema.apply((variable,))
    variable.assign(final_value)
    ema.apply((variable,))

    self.assertAllClose(ema.averaged_variables[0], final_value)

  def test_cross_replica_context_raises_error(self):
    ema = exponential_moving_average.ExponentialMovingAverage(decay=0.0)

    with self.assertRaisesRegex(
        NotImplementedError, 'Cross-replica context version not implemented.'):
      with tf.distribute.MirroredStrategy().scope():
        variable = tf.Variable(2.0)
        ema.apply((variable,))

  def test_mirrored_strategy_replica_context_runs(self):
    ema = exponential_moving_average.ExponentialMovingAverage(decay=0.5)
    strategy = tf.distribute.MirroredStrategy()

    def apply_to_ema(variable):
      ema.apply((variable,))

    with strategy.scope():
      variable = tf.Variable(2.0)
      strategy.run(apply_to_ema, (variable,))

    self.assertAllClose(ema.averaged_variables[0], variable.read_value())


if __name__ == '__main__':
  tf.test.main()
