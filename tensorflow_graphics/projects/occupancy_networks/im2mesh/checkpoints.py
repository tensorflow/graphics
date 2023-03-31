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
""" NO COMMENT NOW"""

import os

import tensorflow as tf
import numpy as np


class CheckpointIO(object):
  """ CheckpointIO class.

  It handles saving and loading checkpoints.

  Args:
      model (tf.keras.Model): model saved with the checkpoints
      optimizer (tf.keras.optimizers): optimizer saved with the checkpoints
      model_selection_sign (int): parameter needed for initializing
        metric_val_best
      checkpoint_dir (str): path where checkpoints are saved
  """

  def __init__(self, model, optimizer, model_selection_sign=1,
               checkpoint_dir="./chkpts"):
    self.ckpt = tf.train.Checkpoint(
        model=model, optimizer=optimizer, epoch_it=tf.Variable(
            -1, dtype=tf.int64),
        it=tf.Variable(-1, dtype=tf.int64),
        metric_val_best=tf.Variable(-model_selection_sign * np.inf,
                                    dtype=tf.float32))

    self.checkpoint_dir = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

  def save(self, filename, epoch_it, it, loss_val_best):
    """ Saves the current module dictionary.

    Args:
        filename (str): name of output file
        epoch_it (tf.Variable): epoch saved
        it (tf.Variable): iteration saved
        loss_val_best(tf.Variable): metric_val_best saved
    """
    if not os.path.isabs(filename):
      filename = os.path.join(self.checkpoint_dir, filename)

    self.ckpt.epoch_it.assign(epoch_it)
    self.ckpt.it.assign(it)
    self.ckpt.metric_val_best.assign(loss_val_best)
    self.ckpt.save(filename)

  def load(self, filename=None):
    '''Loads a module dictionary from local file or url.

    Args:
        filename (str): name of saved module dictionary
    '''
    if filename is not None:
      if not os.path.isabs(filename):
        filename = os.path.join(self.checkpoint_dir, filename)
        filename = tf.train.latest_checkpoint(filename)
    else:
      filename = tf.train.latest_checkpoint(self.checkpoint_dir)

    if filename is not None:
      print(filename)
      print("=> Loading checkpoint from local file...")
      self.ckpt.restore(filename)
    else:
      raise FileExistsError
