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
"""Gather all the shapenet models in the Corenet dataset."""

import os
import pickle
import subprocess

from absl import app
from absl import flags
from google3.pyglib import gfile


FLAGS = flags.FLAGS
flags.DEFINE_string('tfrecords_dir', '', 'Path to tfrecord files.')
flags.DEFINE_string('split', 'train*.tfrecord', 'Split')


def main(_):
  models_data_filename = 'models_per_split.pkl'

  with gfile.Open(models_data_filename, 'rb') as filename:
    models_data_per_class = pickle.load(filename)

  path_prefix = '/datasets/shapenet/raw/'
  local_prefix = '/occluded_primitives/meshes/'
  for split in ['val', 'train', 'test']:
    for class_name in models_data_per_class[split]:
      for model in models_data_per_class[split][class_name]:
        mesh_file = os.path.join(path_prefix, class_name, model, 'models',
                                 'model_normalized.obj')
        local_dir = os.path.join(local_prefix, split, class_name, model)
        if not os.path.exists(local_dir):
          os.makedirs(local_dir)
        print(mesh_file)
        res = subprocess.call(['fileutil', 'cp', mesh_file, local_dir])
        print(res)


if __name__ == '__main__':
  app.run(main)
