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
"""Dataset implementations."""
from os import path

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def get_dataset(split, hparams):
  return dataset_dict[hparams.dataset](split, hparams)


def amass(split, hparams):
  """Construct an AMASS data loader."""

  def _input_fn(params):  # pylint: disable=unused-argument
    # Dataset constants.
    n_bbox = 100000
    n_surf = 100000
    n_points = n_bbox + n_surf
    n_vert = 6890
    n_frames = 1
    # Parse parameters for global configurations.
    n_dims = hparams.n_dims
    data_dir = hparams.data_dir
    sample_bbox = hparams.sample_bbox
    sample_surf = hparams.sample_surf
    batch_size = hparams.batch_size
    subject = hparams.subject
    motion = hparams.motion
    n_parts = hparams.n_parts

    def _parse_tfrecord(serialized_example):
      fs = tf.parse_single_example(
          serialized_example,
          features={
              'point':
                  tf.FixedLenFeature([n_frames * n_points * n_dims],
                                     tf.float32),
              'label':
                  tf.FixedLenFeature([n_frames * n_points * 1], tf.float32),
              'vert':
                  tf.FixedLenFeature([n_frames * n_vert * n_dims], tf.float32),
              'weight':
                  tf.FixedLenFeature([n_frames * n_vert * n_parts], tf.float32),
              'transform':
                  tf.FixedLenFeature(
                      [n_frames * n_parts * (n_dims + 1) * (n_dims + 1)],
                      tf.float32),
              'joint':
                  tf.FixedLenFeature([n_frames * n_parts * n_dims], tf.float32),
              'name':
                  tf.FixedLenFeature([], tf.string),
          })
      fs['point'] = tf.reshape(fs['point'], [n_frames, n_points, n_dims])
      fs['label'] = tf.reshape(fs['label'], [n_frames, n_points, 1])
      fs['vert'] = tf.reshape(fs['vert'], [n_frames, n_vert, n_dims])
      fs['weight'] = tf.reshape(fs['weight'], [n_frames, n_vert, n_parts])
      fs['transform'] = tf.reshape(fs['transform'],
                                   [n_frames, n_parts, n_dims + 1, n_dims + 1])
      fs['joint'] = tf.reshape(fs['joint'], [n_frames, n_parts, n_dims])
      return fs

    def _sample_frame_points(fs):
      feature = {}
      for k, v in fs.items():
        feature[k] = v
      points = feature['point'][0]
      labels = feature['label'][0]
      sample_points = []
      sample_labels = []
      if sample_bbox > 0:
        indices_bbox = tf.random.uniform([sample_bbox],
                                         minval=0,
                                         maxval=n_bbox,
                                         dtype=tf.int32)
        bbox_samples = tf.gather(points[:n_bbox], indices_bbox, axis=0)
        bbox_labels = tf.gather(labels[:n_bbox], indices_bbox, axis=0)
        sample_points.append(bbox_samples)
        sample_labels.append(bbox_labels)
      if sample_surf > 0:
        indices_surf = tf.random.uniform([sample_surf],
                                         minval=0,
                                         maxval=n_surf,
                                         dtype=tf.int32)
        surf_samples = tf.gather(
            points[n_bbox:n_bbox + n_surf], indices_surf, axis=0)
        surf_labels = tf.gather(
            labels[n_bbox:n_bbox + n_surf], indices_surf, axis=0)
        sample_points.append(surf_samples)
        sample_labels.append(surf_labels)
      points = tf.concat(sample_points, axis=0)
      point_labels = tf.concat(sample_labels, axis=0)
      feature['point'] = tf.expand_dims(points, axis=0)
      feature['label'] = tf.expand_dims(point_labels, axis=0)
      return feature

    def _sample_eval_points(fs):
      feature = {}

      feature['transform'] = fs['transform']

      feature['points'] = fs['point'][:, :n_bbox]
      feature['labels'] = fs['label'][:, :n_bbox]

      feature['name'] = fs['name']
      feature['vert'] = fs['vert']
      feature['weight'] = fs['weight']
      feature['joint'] = fs['joint']

      return feature

    data_split = 'train'
    all_motions = list(x for x in range(10))
    if split == 'train':
      file_pattern = [
          path.join(data_dir,
                    '{0}-{1:02d}-{2:02d}-*'.format(data_split, subject, x))
          for x in all_motions if x != motion
      ]
    else:
      file_pattern = [
          path.join(data_dir,
                    '{0}-{1:02d}-{2:02d}-*'.format(data_split, subject, motion))
      ]
    data_files = tf.gfile.Glob(file_pattern)
    if not data_files:
      raise IOError('{} did not match any files'.format(file_pattern))
    filenames = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    data = filenames.interleave(
        lambda x: tf.data.TFRecordDataset([x]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    data = data.map(
        _parse_tfrecord,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
    if split == 'train':
      data = data.map(
          _sample_frame_points,
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      data = data.map(
          _sample_eval_points, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if split == 'train':
      data = data.shuffle(int(batch_size * 2.5)).repeat(-1)
    else:
      batch_size = 1

    return data.batch(
        batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

  return _input_fn


dataset_dict = {
    'amass': amass,
}
