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
"""Test helpers for the transformation module."""

import itertools
import math

import numpy as np
from scipy import stats
import tensorflow.compat.v2 as tf

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


def generate_preset_test_translations(dimensions=3):
  """Generates a set of translations."""
  permutations = itertools.product([0.1, -0.2, 0.5, 0.7, 0.4, -0.1],
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


def generate_preset_test_dual_quaternions():
  """Generates pre-set test quaternions."""
  angles = generate_preset_test_euler_angles()
  preset_quaternion_real = quaternion.from_euler(angles)

  translations = generate_preset_test_translations()
  translations = np.concatenate(
      (translations / 2.0, np.zeros((np.ma.size(translations, 0), 1))), axis=1)
  preset_quaternion_translation = tf.convert_to_tensor(value=translations)

  preset_quaternion_dual = quaternion.multiply(preset_quaternion_translation,
                                               preset_quaternion_real)

  preset_dual_quaternion = tf.concat(
      (preset_quaternion_real, preset_quaternion_dual), axis=-1)

  return preset_dual_quaternion


def generate_random_test_dual_quaternions():
  """Generates random test dual quaternions."""
  angles = generate_random_test_euler_angles()
  random_quaternion_real = quaternion.from_euler(angles)

  min_translation = -3.0
  max_translation = 3.0
  translations = np.random.uniform(min_translation, max_translation,
                                   angles.shape)

  translations_quaternion_shape = np.asarray(translations.shape)
  translations_quaternion_shape[-1] = 1
  translations = np.concatenate(
      (translations / 2.0, np.zeros(translations_quaternion_shape)), axis=-1)

  random_quaternion_translation = tf.convert_to_tensor(value=translations)

  random_quaternion_dual = quaternion.multiply(random_quaternion_translation,
                                               random_quaternion_real)

  random_dual_quaternion = tf.concat(
      (random_quaternion_real, random_quaternion_dual), axis=-1)

  return random_dual_quaternion


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


def generate_random_test_lbs_blend():
  """Generates random test for the linear blend skinning blend function."""
  tensor_dimensions = np.random.randint(3)
  tensor_shape = np.random.randint(1, 10, size=(tensor_dimensions)).tolist()
  random_points = np.random.uniform(size=tensor_shape + [3])
  num_weights = np.random.randint(2, 10)
  random_weights = np.random.uniform(size=tensor_shape + [num_weights])
  random_weights /= np.sum(random_weights, axis=-1, keepdims=True)

  random_rotations = np.array(
      [stats.special_ortho_group.rvs(3) for _ in range(num_weights)])
  random_rotations = np.reshape(random_rotations, [num_weights, 3, 3])
  random_translations = np.random.uniform(size=[num_weights, 3])
  return random_points, random_weights, random_rotations, random_translations


def generate_preset_test_lbs_blend():
  """Generates preset test for the linear blend skinning blend function."""
  points = np.array([[[1.0, 0.0, 0.0], [0.1, 0.2, 0.5]],
                     [[0.0, 1.0, 0.0], [0.3, -0.5, 0.2]],
                     [[-0.3, 0.1, 0.3], [0.1, -0.9, -0.4]]])
  weights = np.array([[[0.0, 1.0, 0.0, 0.0], [0.4, 0.2, 0.3, 0.1]],
                      [[0.6, 0.0, 0.4, 0.0], [0.2, 0.2, 0.1, 0.5]],
                      [[0.0, 0.1, 0.0, 0.9], [0.1, 0.2, 0.3, 0.4]]])
  rotations = np.array(
      [[[[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]],
        [[0.36, 0.48, -0.8],
         [-0.8, 0.60, 0.00],
         [0.48, 0.64, 0.60]],
        [[0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0]]],
       [[[-0.41554751, -0.42205085, -0.80572535],
         [0.08028719, -0.89939186, 0.42970716],
         [-0.9060211, 0.11387432, 0.40762533]],
        [[-0.05240625, -0.24389111, 0.96838562],
         [0.99123384, -0.13047444, 0.02078231],
         [0.12128095, 0.96098572, 0.2485908]],
        [[-0.32722936, -0.06793413, -0.94249981],
         [-0.70574479, 0.68082693, 0.19595657],
         [0.62836712, 0.72928708, -0.27073072]],
        [[-0.22601332, -0.95393284, 0.19730719],
         [-0.01189659, 0.20523618, 0.97864017],
         [-0.97405157, 0.21883843, -0.05773466]]]])  # pyformat: disable
  translations = np.array(
      [[[0.1, -0.2, 0.5],
        [-0.2, 0.7, 0.7],
        [0.8, -0.2, 0.4],
        [-0.1, 0.2, -0.3]],
       [[0.5, 0.6, 0.9],
        [-0.1, -0.3, -0.7],
        [0.4, -0.2, 0.8],
        [0.7, 0.8, -0.4]]])  # pyformat: disable
  blended_points = np.array([[[[0.16, -0.1, 1.18], [0.3864, 0.148, 0.7352]],
                              [[0.38, 0.4, 0.86], [-0.2184, 0.152, 0.0088]],
                              [[-0.05, 0.01, -0.46], [-0.3152, -0.004,
                                                      -0.1136]]],
                             [[[-0.15240625, 0.69123384, -0.57871905],
                               [0.07776242, 0.33587402, 0.55386645]],
                              [[0.17959584, 0.01269566, 1.22003942],
                               [0.71406514, 0.6187734, -0.43794053]],
                              [[0.67662743, 0.94549789, -0.14946982],
                               [0.88587099, -0.09324637, -0.45012815]]]])

  return points, weights, rotations, translations, blended_points
