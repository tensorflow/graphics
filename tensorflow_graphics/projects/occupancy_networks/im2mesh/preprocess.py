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

import tensorflow as tf
import numpy as np

from im2mesh import config
from im2mesh.checkpoints import CheckpointIO


class PSGNPreprocessor:
  ''' Point Set Generation Networks (PSGN) preprocessor class.

  Args:
      cfg_path (str): path to config file
      pointcloud_n (int): number of output points
      dataset (dataset): dataset
      model_file (str): model file
  '''

  def __init__(self,
               cfg_path,
               pointcloud_n,
               dataset=None,
               model_file=None):
    self.cfg = config.load_config(cfg_path, 'configs/default.yaml')
    self.pointcloud_n = pointcloud_n
    self.dataset = dataset
    self.model = config.get_model(self.cfg, dataset)

    # Output directory of psgn model
    out_dir = self.cfg['training']['out_dir']
    # If model_file not specified, use the one from psgn model
    if model_file is None:
      model_file = self.cfg['test']['model_file']
    # Load model
    self.checkpoint_io = CheckpointIO(model=model, checkpoint_dir=out_dir)
    self.checkpoint_io.load(model_file)

  def __call__(self, inputs):
    points = self.model(inputs, training=False)

    batch_size = points.shape[0]
    t = points.shape[1]

    # Subsample points if necessary
    if t != self.pointcloud_n:
      idx = np.random.randint(low=0,
                              high=t,
                              size=(batch_size, self.pointcloud_n))
      idx = tf.convert_to_tensor(idx[:, :, None])
      idx = tf.broadcast_to(
          idx, shape=[batch_size, self.pointcloud_n, 3])
      points = tf.gather(points, indices=idx, axis=None, batch_dims=1)

    return points
