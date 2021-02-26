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
"""Logger class."""

import collections
import os
import tensorflow as tf


class Logger:
  """Logger class."""

  def __init__(self,
               logdir,
               name,
               xmanager_work_unit,
               xmanager_metric,
               save_loss_tensorboard_frequency,
               print_loss_frequency):

    self.logdir = logdir
    tensorboard_dir = os.path.join(logdir, f'tb_{name}')
    self.name = name
    self.summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    self.losses = collections.defaultdict(list)
    self.save_loss_tensorboard_frequency = save_loss_tensorboard_frequency
    self.print_loss_frequency = print_loss_frequency
    self.xmanager = {}
    if xmanager_metric:
      for metric in xmanager_metric.split(','):
        self.xmanager[metric] = xmanager_work_unit.get_measurement_series(
            label=metric)

  def record_scalar(self, name, scalar, step):
    tf.summary.scalar(name, scalar, step)
    if name in self.xmanager:
      step = step.numpy() if isinstance(step, tf.Tensor) else step
      self.xmanager[name].create_measurement(
          objective_value=float(scalar), step=int(step))

  def record_dictionary_scalars(self, prefix, scalar_dict, step):
    for key, val in scalar_dict.items():
      name = f'{prefix}{key}'
      if isinstance(val, tf.Tensor):
        val = val.numpy()
      self.record_scalar(name, val, step)
    tf.summary.flush()

  def record_losses(self, prefix, losses, step):
    for loss_name, val in losses.items():
      if isinstance(val, tf.Tensor):
        val = val.numpy()
      self.losses[loss_name].append(val)
      if step % self.save_loss_tensorboard_frequency == 0:
        name = f'{prefix}{loss_name}'
        self.record_scalar(name, val, step)
      if step % self.print_loss_frequency == 0:
        print(f'Step: {step}, Loss {loss_name}: {val}')

  def record_losses_epoch(self, prefix, step):
    for loss_name, val in self.losses.items():
      name = f'{prefix}epoch_{loss_name}'
      self.record_scalar(name, sum(val) / len(val), step)

  def reset_losses(self):
    self.losses = collections.defaultdict(list)
