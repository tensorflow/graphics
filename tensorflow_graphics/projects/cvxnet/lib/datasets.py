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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path

import tensorflow.compat.v1 as tf


def get_dataset(data_name, split, args):
  return dataset_dict[data_name](split, args)


def shapenet(split, args):
  """ShapeNet Dataset.

  Args:
    split: string, the split of the dataset, either "train" or "test".
    args: tf.app.flags.FLAGS, configurations.

  Returns:
    dataset: tf.data.Dataset, the shapenet dataset.
  """
  total_points = 100000
  data_dir = args.data_dir
  sample_bbx = args.sample_bbx
  if split != "train":
    sample_bbx = total_points
  sample_surf = args.sample_surf
  if split != "train":
    sample_surf = 0
  image_h = args.image_h
  image_w = args.image_w
  image_d = args.image_d
  n_views = args.n_views
  depth_h = args.depth_h
  depth_w = args.depth_w
  depth_d = args.depth_d
  batch_size = args.batch_size if split == "train" else 1
  dims = args.dims

  def _parser(example):
    fs = tf.parse_single_example(
        example,
        features={
            "rgb":
                tf.FixedLenFeature([n_views * image_h * image_w * image_d],
                                   tf.float32),
            "depth":
                tf.FixedLenFeature([depth_d * depth_h * depth_w], tf.float32),
            "bbox_samples":
                tf.FixedLenFeature([total_points * (dims + 1)], tf.float32),
            "surf_samples":
                tf.FixedLenFeature([total_points * (dims + 1)], tf.float32),
            "name":
                tf.FixedLenFeature([], tf.string),
        })
    fs["rgb"] = tf.reshape(fs["rgb"], [n_views, image_h, image_w, image_d])
    fs["depth"] = tf.reshape(fs["depth"], [depth_d, depth_h, depth_w, 1])
    fs["bbox_samples"] = tf.reshape(fs["bbox_samples"],
                                    [total_points, dims + 1])
    fs["surf_samples"] = tf.reshape(fs["surf_samples"],
                                    [total_points, dims + 1])
    return fs

  def _sampler(example):
    image = tf.gather(
        example["rgb"],
        tf.random.uniform((),
                          minval=0,
                          maxval=n_views if split == "train" else 1,
                          dtype=tf.int32),
        axis=0)
    image = tf.image.resize_bilinear(tf.expand_dims(image, axis=0), [224, 224])

    depth = example["depth"] / 1000.

    sample_points = []
    sample_labels = []

    if sample_bbx > 0:
      if split == "train":
        indices_bbx = tf.random.uniform([sample_bbx],
                                        minval=0,
                                        maxval=total_points,
                                        dtype=tf.int32)
        bbx_samples = tf.gather(example["bbox_samples"], indices_bbx, axis=0)
      else:
        bbx_samples = example["bbox_samples"]
      bbx_points, bbx_labels = tf.split(bbx_samples, [3, 1], axis=-1)
      sample_points.append(bbx_points)
      sample_labels.append(bbx_labels)

    if sample_surf > 0:
      indices_surf = tf.random.uniform([sample_surf],
                                       minval=0,
                                       maxval=total_points,
                                       dtype=tf.int32)
      surf_samples = tf.gather(example["surf_samples"], indices_surf, axis=0)
      surf_points, surf_labels = tf.split(surf_samples, [3, 1], axis=-1)
      sample_points.append(surf_points)
      sample_labels.append(surf_labels)

    points = tf.concat(sample_points, axis=0)
    point_labels = tf.cast(tf.concat(sample_labels, axis=0) <= 0., tf.float32)

    image = tf.reshape(image, [224, 224, image_d])
    depth = tf.reshape(depth, [depth_d, depth_h, depth_w])
    depth = tf.transpose(depth, [1, 2, 0])
    points = tf.reshape(points, [sample_bbx + sample_surf, 3])
    point_labels = tf.reshape(point_labels, [sample_bbx + sample_surf, 1])

    return {
        "image": image,
        "depth": depth,
        "point": points,
        "point_label": point_labels,
        "name": example["name"],
    }

  data_pattern = path.join(data_dir, "{}-{}-*".format(args.obj_class, split))
  data_files = tf.gfile.Glob(data_pattern)
  if not data_files:
    raise ValueError("{} did not match any files".format(data_pattern))
  file_count = len(data_files)
  filenames = tf.data.Dataset.list_files(data_pattern, shuffle=True)
  data = filenames.interleave(
      lambda x: tf.data.TFRecordDataset([x]),
      cycle_length=file_count,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  data = data.map(_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  data = data.map(_sampler, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if split == "train":
    data = data.shuffle(batch_size * 5).repeat(-1)

  return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


dataset_dict = {
    "shapenet": shapenet,
}
