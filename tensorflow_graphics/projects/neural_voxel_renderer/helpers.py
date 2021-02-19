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
"""Helper functions for neural voxel renderer +."""
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.math.interpolation import trilinear
from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.rendering.volumetric import emission_absorption


X_AXIS = np.array((1., 0., 0.), dtype=np.float32)
Z_AXIS = np.array((0., 0., 1.), dtype=np.float32)

PI = np.array((-np.pi,), dtype=np.float32)
PI_2 = np.array((-np.pi/2.,), dtype=np.float32)
ZERO = np.array((0.,), dtype=np.float32)
MIRROR_X_MATRIX = np.eye(3, dtype=np.float32)
MIRROR_X_MATRIX[0, 0] = -1

# Cube that encloses all the shapes in the dataset (world coordinates)
CUBE_BOX_DIM = 1.1528184

# The bottom part of all shapes in world coordinates
OBJECT_BOTTOM = -1.1167929


def generate_ground_image(height,
                          width,
                          focal,
                          principal_point,
                          camera_rotation_matrix,
                          camera_translation_vector,
                          ground_color=(0.43, 0.43, 0.8)):
  """Generate an image depicting only the ground."""
  batch_size = camera_rotation_matrix.shape[0]
  background_image = np.ones((batch_size, height, width, 1, 1),
                             dtype=np.float32)
  background_image[:, -1, ...] = 0  # Zero the bottom line for proper sampling

  # The projection of the ground depends on the top right corner (approximation)
  plane_point_np = np.tile(np.array([[3.077984, 2.905388, 0.]],
                                    dtype=np.float32), (batch_size, 1))
  plane_point_rotated = rotation_matrix_3d.rotate(plane_point_np,
                                                  camera_rotation_matrix)
  plane_point_translated = plane_point_rotated + camera_translation_vector
  plane_point2d = \
    perspective.project(plane_point_translated, focal, principal_point)
  _, y = tf.split(plane_point2d, [1, 1], axis=-1)
  sfactor = height/y
  helper_matrix1 = np.tile(np.array([[[1, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 0]]]), (batch_size, 1, 1))
  helper_matrix2 = np.tile(np.array([[[0, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]]), (batch_size, 1, 1))
  transformation_matrix = tf.multiply(tf.expand_dims(sfactor, -1),
                                      helper_matrix1) + helper_matrix2
  plane_points = grid.generate((0., 0., 0.),
                               (float(height), float(width), 0.),
                               (height, width, 1))
  plane_points = tf.reshape(plane_points, [-1, 3])
  transf_plane_points = tf.matmul(transformation_matrix,
                                  plane_points,
                                  transpose_b=True)
  interpolated_points = \
    trilinear.interpolate(background_image,
                          tf.linalg.matrix_transpose(transf_plane_points))
  ground_alpha = (1- tf.reshape(interpolated_points,
                                [batch_size, height, width, 1]))
  ground_image = tf.ones((batch_size, height, width, 3))*ground_color
  return ground_image, ground_alpha


def sampling_points_to_voxel_index(sampling_points, voxel_size):
  """Transforms the sampling points from [-1, 1] to [0, voxel_size]."""
  voxel_size = tf.convert_to_tensor(value=voxel_size)
  max_size = tf.cast(voxel_size - 1, sampling_points.dtype)
  return 0.5 * ((sampling_points + 1) * max_size)


def sampling_points_from_3d_grid(grid_size, dtype=tf.float32):
  """Returns a tensor of shape `[M, 3]`, with M the number of sampling points."""
  sampling_points = grid.generate((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
                                  grid_size)
  sampling_points = tf.cast(sampling_points, dtype)
  return tf.reshape(sampling_points, [-1, 3])


def sampling_points_from_frustum(height,
                                 width,
                                 focal,
                                 principal_point,
                                 depth_min=0.0,
                                 depth_max=5.,
                                 frustum_size=(256, 256, 256)):
  """Generates samples from a camera frustum."""

  # ------------------ Get the rays from the camera ----------------------------
  sampling_points = grid.generate((0., 0.),
                                  (float(width)-1, float(height)-1),
                                  (frustum_size[0], frustum_size[1]))
  sampling_points = tf.reshape(sampling_points, [-1, 2])  # [h*w, 2]
  rays = perspective.ray(sampling_points, focal, principal_point)  # [h*w, 3]

  # ------------------ Extract a volume in front of the camera -----------------
  depth_tensor = grid.generate((depth_min,), (depth_max,), (frustum_size[2],))
  sampling_volume = tf.multiply(tf.expand_dims(rays, axis=-1),
                                tf.transpose(depth_tensor))  # [h*w, 3, dstep]
  sampling_volume = tf.transpose(sampling_volume, [0, 2, 1])  # [h*w, dstep, 3]
  sampling_volume = tf.reshape(sampling_volume, [-1, 3])  # [h*w*dstep, 3]
  return sampling_volume


def place_frustum_sampling_points_at_blender_camera(sampling_points,
                                                    camera_rotation_matrix,
                                                    camera_translation_vector):
  """Transform the TF-Graphics frustum points to the blender camera position."""
  blender_rotation_matrix = rotation_matrix_3d.from_axis_angle(X_AXIS, -PI)
  camera_translation_vector = tf.matmul(blender_rotation_matrix,
                                        camera_translation_vector)

  # The original frustum points are mirrored in the x axis and rotated 270
  mirror_x_matrix = tf.convert_to_tensor(MIRROR_X_MATRIX)
  up_rotation_matrix = rotation_matrix_3d.from_axis_angle(Z_AXIS, 3*PI_2)
  local_camera_transform = tf.matmul(mirror_x_matrix, up_rotation_matrix)

  sampling_points = tf.matmul(local_camera_transform,
                              sampling_points,
                              transpose_b=True)
  sampling_points = \
    tf.matmul(tf.linalg.matrix_transpose(camera_rotation_matrix),
              sampling_points+camera_translation_vector)
  return tf.linalg.matrix_transpose(sampling_points)  # [h*w*dstep, 3]


def transform_volume(voxels, transformation_matrix, voxel_size=(128, 128, 128)):
  """Apply a transformation to the input voxels."""
  voxels = tf.convert_to_tensor(voxels)
  volume_sampling = sampling_points_from_3d_grid(voxel_size)
  volume_sampling = tf.matmul(transformation_matrix,
                              tf.transpose(a=volume_sampling))
  volume_sampling = tf.cast(tf.linalg.matrix_transpose(volume_sampling),
                            tf.float32)
  volume_sampling = sampling_points_to_voxel_index(volume_sampling, voxel_size)
  interpolated_voxels = trilinear.interpolate(voxels, volume_sampling)
  return tf.reshape(interpolated_voxels, tf.shape(voxels))


def object_rotation_in_blender_world(voxels, object_rotation):
  """Rotate the voxels as in blender world."""
  euler_angles = np.array([0, 0, 1], dtype=np.float32)*np.deg2rad(90)
  object_correction_matrix = rotation_matrix_3d.from_euler(euler_angles)
  euler_angles = np.array([0, 1, 0], dtype=np.float32)*(-object_rotation)
  object_rotation_matrix = rotation_matrix_3d.from_euler(euler_angles)
  euler_angles_blender = np.array([1, 0, 0], dtype=np.float32)*np.deg2rad(-90)
  blender_object_correction_matrix = \
    rotation_matrix_3d.from_euler(euler_angles_blender)
  transformation_matrix = tf.matmul(tf.matmul(object_correction_matrix,
                                              object_rotation_matrix),
                                    blender_object_correction_matrix)

  return transform_volume(voxels, transformation_matrix)


def render_voxels_from_blender_camera(voxels,
                                      object_rotation,
                                      object_translation,
                                      height,
                                      width,
                                      focal,
                                      principal_point,
                                      camera_rotation_matrix,
                                      camera_translation_vector,
                                      frustum_size=(256, 256, 512),
                                      absorption_factor=0.1,
                                      cell_size=1.0,
                                      depth_min=0.0,
                                      depth_max=5.0):
  """Renders the voxels according to their position in the world."""
  batch_size = voxels.shape[0]
  voxel_size = voxels.shape[1]
  sampling_volume = sampling_points_from_frustum(height,
                                                 width,
                                                 focal,
                                                 principal_point,
                                                 depth_min=depth_min,
                                                 depth_max=depth_max,
                                                 frustum_size=frustum_size)
  sampling_volume = \
    place_frustum_sampling_points_at_blender_camera(sampling_volume,
                                                    camera_rotation_matrix,
                                                    camera_translation_vector)

  interpolated_voxels = \
    object_rotation_in_blender_world(voxels, object_rotation)

  # Adjust the camera (translate the camera instead of the object)
  sampling_volume = sampling_volume - object_translation
  sampling_volume = sampling_volume/CUBE_BOX_DIM
  sampling_volume = sampling_points_to_voxel_index(sampling_volume, voxel_size)

  camera_voxels = trilinear.interpolate(interpolated_voxels, sampling_volume)
  camera_voxels = tf.reshape(camera_voxels,
                             [batch_size] + list(frustum_size) + [4])
  voxel_image = emission_absorption.render(camera_voxels,
                                           absorption_factor=absorption_factor,
                                           cell_size=cell_size)
  return voxel_image


def object_to_world(voxels,
                    euler_angles_x,
                    euler_angles_y,
                    translation_vector,
                    target_volume_size=(128, 128, 128)):
  """Apply the transformations to the voxels and place them in world coords."""
  scale_factor = 1.82  # object to world voxel space scale factor

  translation_vector = tf.expand_dims(translation_vector, axis=-1)

  sampling_points = tf.cast(sampling_points_from_3d_grid(target_volume_size),
                            tf.float32)  # 128^3 X 3
  transf_matrix_x = rotation_matrix_3d.from_euler(euler_angles_x)  # [B, 3, 3]
  transf_matrix_y = rotation_matrix_3d.from_euler(euler_angles_y)  # [B, 3, 3]
  transf_matrix = tf.matmul(transf_matrix_x, transf_matrix_y)  # [B, 3, 3]
  transf_matrix = transf_matrix*scale_factor  # [B, 3, 3]
  sampling_points = tf.matmul(transf_matrix,
                              tf.transpose(sampling_points))  # [B, 3, N]
  translation_vector = tf.matmul(transf_matrix, translation_vector)  # [B, 3, 1]
  sampling_points = sampling_points - translation_vector
  sampling_points = tf.linalg.matrix_transpose(sampling_points)
  sampling_points = sampling_points_to_voxel_index(sampling_points,
                                                   target_volume_size)
  sampling_points = tf.cast(sampling_points, tf.float32)
  interpolated_points = trilinear.interpolate(voxels, sampling_points)
  interpolated_voxels = tf.reshape(interpolated_points,
                                   [-1] + list(target_volume_size)+[4])

  return interpolated_voxels
