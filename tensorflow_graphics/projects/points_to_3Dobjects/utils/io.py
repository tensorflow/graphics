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
"""Input/Output utilities."""
import re
import tensorflow as tf


def expand_rio_pattern(rio_pattern):
  def format_shards(m):
    return '{}-?????-of-{:0>5}{}'.format(*m.groups())
  rio_pattern = re.sub(r'^([^@]+)@(\d+)([^@]+)$', format_shards, rio_pattern)
  return rio_pattern


def tfrecords_to_dataset_tf2(tfrecords_pattern,
                             mapping_func,
                             batch_size,
                             buffer_size=5000,
                             shuffle=True):
  """Generates a TF Dataset from a rio pattern."""
  with tf.name_scope('Input/'):
    tfrecords_pattern = expand_rio_pattern(tfrecords_pattern)
    dataset = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=shuffle)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(mapping_func)
    dataset = dataset.batch(batch_size)
    return dataset


def get_dataset(tfrecords_pattern,
                mapping_func,
                buffer_size=500,
                shuffle=True,
                cycle_length=16):
  """Generates a TF Dataset from a rio pattern."""
  with tf.name_scope('Input/'):
    tfrecords_pattern = expand_rio_pattern(tfrecords_pattern)
    dataset = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=shuffle)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=cycle_length)
    # for d in dataset.take(1):
    #   mapping_func(d)
    if shuffle:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.map(mapping_func)
    return dataset
