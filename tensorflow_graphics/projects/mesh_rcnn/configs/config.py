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
"""Configuration for Mesh R-CNN shape branches."""

import json
import os

import tensorflow as tf


class MeshRCNNConfig:
  """Base class to configurate a Mesh R-CNN model for training or testing."""

  def __init__(self, path_to_json=None):
    """Constructor of the config class.

    You can override this config by passing a json configuration as an argument.

    Args:
      path_to_json: `String` describing a path to a json file containing a
        Mesh R-CNN configuration. Have a look at `mesh_r_cnn_base-config.json`
        for an example.
    """
    self.path_to_config_file = os.path.join(os.path.dirname(__file__),
                                            'mesh_r_cnn_base-config.json')

    if not path_to_json is None:
      if not os.path.exists(path_to_json):
        raise ValueError(f'Cannot find {path_to_json}.')

      self.path_to_config_file = os.path.abspath(path_to_json)

    self.parse_config_file(self.path_to_config_file)

  def parse_config_file(self, path_to_config_file):
    """Parses a json configuration file and sets properties of this class."""

    with tf.io.gfile.GFile(path_to_config_file, mode='r') as config_file:
      config = json.load(config_file)

    voxel_config = config['voxel_prediction']
    mesh_config = config['mesh_refinement']

    self.cubify_threshold = config['cubify_threshold']
    self.mesh_loss_sample_size_gt = config['mesh_loss_sample_size_gt']
    self.mesh_loss_sample_size_pred = config['mesh_loss_sample_size_pred']

    self.voxel_prediction_num_convs = voxel_config['num_convs']
    self.voxel_prediction_latent_dim = voxel_config['latent_dim']
    self.voxel_prediction_out_depth = voxel_config['output_depth']
    self.voxel_prediction_layer_name = voxel_config['layer_name']

    self.mesh_refinement_num_stages = mesh_config['num_stages']
    self.mesh_refinement_num_gconvs = mesh_config['num_gconvs_per_stage']
    self.mesh_refinement_gconv_dim = mesh_config['gconv_dim']
    self.mesh_refinement_gconv_initializer = mesh_config['gconv_initializer']
    self.mesh_refinement_layer_name = mesh_config['layer_name']

    self.loss_weights = config['loss_weights']

    # ToDo validate data
