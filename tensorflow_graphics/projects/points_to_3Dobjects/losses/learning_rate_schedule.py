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
"""Warmup delayed cosine decay learning rate scheduler."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf


class WarmupDelayedCosineDecay(tf.keras.LearningRateSchedule):
  """A LearningRateSchedule that uses an delayed cosine decay schedule."""

  def __init__(
      self,
      initial_learning_rate,
      constant_learning_rate,
      end_learning_rate,
      warmup_steps,
      start_decay_step,
      decay_steps,
      name=None):
    """Applies delayed cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      constant_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The minimal end learning rate.
      warmup_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      start_decay_step: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The decay start start.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      name: String.  Optional name of the operation.  Defaults to
        'ExponentialDecay'.
    """
    super(WarmupDelayedCosineDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.constant_learning_rate = constant_learning_rate
    self.end_learning_rate = end_learning_rate
    self.warmup_steps = warmup_steps
    self.start_decay_step = start_decay_step
    self.decay_steps = decay_steps
    self.name = name

  def __call__(self, step):
    with tf.ops.name_scope_v2(self.name or "WarmupDelayedCosineDecay") as name:
      constant_learning_rate = tf.ops.convert_to_tensor_v2_with_dispatch(
          self.constant_learning_rate)
      dtype = constant_learning_rate.dtype
      initial_learning_rate = tf.cast(self.initial_learning_rate, dtype)
      end_learning_rate = tf.cast(self.end_learning_rate, dtype)

      def cosine_decay(s, a, b, c, d):
        """Applies cosine decay to given step s.

        Args:
          s: The current step.
          a: The step when to start decay.
          b: The number of decay steps.
          c: The learning rate before the decay.
          d: The learning rate after the decay.

        Returns:
          Decay function applied to given step.
        """
        return tf.cos((s - a) * math.pi/b) * (c - d)/2.0 + (d + c)/ 2.0

      def warmup_phase():
        return cosine_decay(tf.cast(step, dtype),
                            0,
                            tf.cast(self.warmup_steps, dtype),
                            initial_learning_rate, constant_learning_rate)

      def constant_phase():
        return constant_learning_rate

      def decay_phase():
        return cosine_decay(tf.cast(step, dtype),
                            tf.cast(self.start_decay_step, dtype),
                            tf.cast(self.decay_steps, dtype),
                            constant_learning_rate, end_learning_rate)

      def end_phase():
        return end_learning_rate

      return tf.control_flow_ops.case(
          [(step < self.warmup_steps, warmup_phase),
           (step < self.start_decay_step, constant_phase),
           (step < self.start_decay_step + self.decay_steps, decay_phase)],
          default=end_phase, exclusive=False, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "constant_learning_rate": self.constant_learning_rate,
        "end_learning_rate": self.end_learning_rate,
        "warmup_steps": self.warmup_steps,
        "start_decay_step": self.start_decay_step,
        "decay_steps": self.decay_steps,
        "name": self.name
    }
