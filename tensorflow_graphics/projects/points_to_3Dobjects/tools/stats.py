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
"""Create statistsics over corenet dataset."""

import os

from absl import app
from absl import flags

import tensorflow_graphics.projects.points_to_3Dobjects.data_preparation.extract_protos as extract_protos
import tensorflow_graphics.projects.points_to_3Dobjects.utils.io as io

TFRECORDS_DIR = '/usr/local/google/home/engelmann/occluded_primitives/data_stefan/'

FLAGS = flags.FLAGS
flags.DEFINE_string('tfrecords_dir', TFRECORDS_DIR, 'Path to tfrecord files.')
flags.DEFINE_string('split', 'train*.tfrecord', 'Split')


def count_objects():
  """Print statistics over number of objects."""

  tfrecord_path = os.path.join(FLAGS.tfrecords_dir, FLAGS.split)
  mapping_function = extract_protos.decode_bytes_multiple
  buffer_size, shuffle, cycle_length = 500, False, 1
  dataset = io.get_dataset(tfrecord_path, mapping_function,
                           buffer_size, shuffle, cycle_length)
  i = 0
  my_dict = {}
  my_dict[3001627] = set()
  my_dict[4256520] = set()
  my_dict[4379243] = set()
  my_dict[2876657] = set()
  my_dict[2880940] = set()
  my_dict[3797390] = set()
  for d in dataset:
    classes = d['classes']
    mesh_names = d['mesh_names']
    for i, m in enumerate(list(mesh_names.numpy())):
      my_dict[list(classes.numpy())[i]].add(m)
    i += 1
  print(f'Chairs: \t {len(my_dict[3001627])}')
  print(f'Sofa: \t {len(my_dict[4256520])}')
  print(f'Table: \t {len(my_dict[4379243])}')
  print(f'Bottle: \t {len(my_dict[2876657])}')
  print(f'Bowl: \t {len(my_dict[2880940])}')
  print(f'Mug: \t {len(my_dict[3797390])}')


def main(_):
  tfrecord_path = os.path.join(FLAGS.tfrecords_dir, FLAGS.split)
  mapping_function = extract_protos.decode_bytes_multiple
  buffer_size, shuffle, cycle_length = 500, False, 1
  dataset = io.get_dataset(tfrecord_path, mapping_function,
                           buffer_size, shuffle, cycle_length)
  i = 0
  for d in dataset:
    i += 1
    print(d['name'])
    if i % 100 == 0:
      print(i)
  print('final:', i)

if __name__ == '__main__':
  app.run(main)
