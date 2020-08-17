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
"""Data augmentation utility functions."""

import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def jitter(points, stddev=0.01, clip=0.05):
  """Randomly jitters points with (clipped) Gaussian noise."""
  assert (stddev > 0), "jitter needs a positive sigma"
  assert (clip > 0), "jitter needs a positive clip"
  noise = tf.random.normal(points.shape, stddev=stddev)
  noise = tf.clip_by_value(noise, -clip, +clip)
  return points + noise


def rotate(points):
  """Randomly rotates a point cloud around the Y axis (UP)."""
  axis = tf.constant([[0, 1, 0]], dtype=tf.float32)  # [1, 3]
  angle = tf.random.uniform((1, 1), minval=0., maxval=2 * np.pi)  # [1, 1]
  matrix = rotation_matrix_3d.from_axis_angle(axis, angle)  # [3, 3]
  return rotation_matrix_3d.rotate(points, matrix)  # [N, 3]
