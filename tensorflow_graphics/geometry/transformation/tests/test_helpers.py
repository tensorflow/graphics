#Copyright 2018 Google LLC
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
"""Test helpers for the transformation module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

import numpy as np
from scipy import stats
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import axis_angle
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_2d
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def generate_preset_test_euler_angles(dimensions=3):
  """Generates a permutation with duplicate of some classic euler angles."""
  permutations = itertools.product(
      [0., np.pi, np.pi / 2., np.pi / 3., np.pi / 4., np.pi / 6.],
      repeat=dimensions)
  return np.array(list(permutations))


def generate_preset_test_rotation_matrices_3d():
  """Generates pre-set test 3d rotation matrices."""
  angles = generate_preset_test_euler_angles()
  preset_rotation_matrix = rotation_matrix_3d.from_euler(angles)
  if tf.executing_eagerly():
    return np.array(preset_rotation_matrix)
  with tf.compat.v1.Session() as sess:
    return np.array(sess.run([preset_rotation_matrix]))


def generate_preset_test_rotation_matrices_2d():
  """Generates pre-set test 2d rotation matrices."""
  angles = generate_preset_test_euler_angles(dimensions=1)
  preset_rotation_matrix = rotation_matrix_2d.from_euler(angles)
  if tf.executing_eagerly():
    return np.array(preset_rotation_matrix)
  with tf.compat.v1.Session() as sess:
    return np.array(sess.run([preset_rotation_matrix]))


def generate_preset_test_axis_angle():
  """Generates pre-set test rotation matrices."""
  angles = generate_preset_test_euler_angles()
  axis, angle = axis_angle.from_euler(angles)
  if tf.executing_eagerly():
    return np.array(axis), np.array(angle)
  with tf.compat.v1.Session() as sess:
    return np.array(sess.run([axis])), np.array(sess.run([angle]))


def generate_preset_test_quaternions():
  """Generates pre-set test quaternions."""
  angles = generate_preset_test_euler_angles()
  preset_quaternion = quaternion.from_euler(angles)
  if tf.executing_eagerly():
    return np.array(preset_quaternion)
  with tf.compat.v1.Session() as sess:
    return np.array(sess.run([preset_quaternion]))


def generate_random_test_euler_angles(dimensions=3,
                                      min_angle=-3. * np.pi,
                                      max_angle=3. * np.pi):
  """Generates random test random Euler angles."""
  tensor_dimensions = np.random.randint(3)
  tensor_tile = np.random.randint(1, 10, tensor_dimensions).tolist()
  return np.random.uniform(min_angle, max_angle, tensor_tile + [dimensions])


def generate_random_test_quaternions(tensor_shape=None):
  """Generates random test quaternions."""
  if tensor_shape is None:
    tensor_dimensions = np.random.randint(low=1, high=3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_dimensions)).tolist()
  u1 = np.random.uniform(0.0, 1.0, tensor_shape)
  u2 = np.random.uniform(0.0, 2.0 * math.pi, tensor_shape)
  u3 = np.random.uniform(0.0, 2.0 * math.pi, tensor_shape)
  a = np.sqrt(1.0 - u1)
  b = np.sqrt(u1)
  return np.stack((a * np.sin(u2),
                   a * np.cos(u2),
                   b * np.sin(u3),
                   b * np.cos(u3)),
                  axis=-1)  # pyformat: disable


def generate_random_test_axis_angle():
  """Generates random test axis-angles."""
  tensor_dimensions = np.random.randint(3)
  tensor_shape = np.random.randint(1, 10, size=(tensor_dimensions)).tolist()
  random_axis = np.random.uniform(size=tensor_shape + [3])
  random_axis /= np.linalg.norm(random_axis, axis=-1, keepdims=True)
  random_angle = np.random.uniform(size=tensor_shape + [1])
  return random_axis, random_angle


def generate_random_test_rotation_matrix_3d():
  """Generates random test 3d rotation matrices."""
  random_matrix = np.array(
      [stats.special_ortho_group.rvs(3) for _ in range(20)])
  return np.reshape(random_matrix, [5, 4, 3, 3])


def generate_random_test_rotation_matrix_2d():
  """Generates random test 2d rotation matrices."""
  random_matrix = np.array(
      [stats.special_ortho_group.rvs(2) for _ in range(20)])
  return np.reshape(random_matrix, [5, 4, 2, 2])
