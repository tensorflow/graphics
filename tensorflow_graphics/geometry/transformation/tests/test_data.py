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
"""Module with test data for transformation tests."""

import numpy as np

ANGLE_0 = np.array((0.,))
ANGLE_45 = np.array((np.pi / 4.,))
ANGLE_90 = np.array((np.pi / 2.,))
ANGLE_180 = np.array((np.pi,))

AXIS_2D_0 = np.array((0., 0.))
AXIS_2D_X = np.array((1., 0.))
AXIS_2D_Y = np.array((0., 1.))


def _rotation_2d_x(angle):
  """Creates a 2d rotation matrix.

  Args:
    angle: The angle.

  Returns:
    The 2d rotation matrix.
  """
  angle = angle.item()
  return np.array(((np.cos(angle), -np.sin(angle)),
                   (np.sin(angle), np.cos(angle))))  # pyformat: disable


MAT_2D_ID = np.eye(2)
MAT_2D_45 = _rotation_2d_x(ANGLE_45)
MAT_2D_90 = _rotation_2d_x(ANGLE_90)
MAT_2D_180 = _rotation_2d_x(ANGLE_180)

AXIS_3D_0 = np.array((0., 0., 0.))
AXIS_3D_X = np.array((1., 0., 0.))
AXIS_3D_Y = np.array((0., 1., 0.))
AXIS_3D_Z = np.array((0., 0., 1.))


def _axis_angle_to_quaternion(axis, angle):
  """Converts an axis-angle representation to a quaternion.

  Args:
    axis: The axis of rotation.
    angle: The angle.

  Returns:
    The quaternion.
  """
  quat = np.zeros(4)
  quat[0:3] = axis * np.sin(0.5 * angle)
  quat[3] = np.cos(0.5 * angle)
  return quat


QUAT_ID = _axis_angle_to_quaternion(AXIS_3D_0, ANGLE_0)
QUAT_X_45 = _axis_angle_to_quaternion(AXIS_3D_X, ANGLE_45)
QUAT_X_90 = _axis_angle_to_quaternion(AXIS_3D_X, ANGLE_90)
QUAT_X_180 = _axis_angle_to_quaternion(AXIS_3D_X, ANGLE_180)
QUAT_Y_45 = _axis_angle_to_quaternion(AXIS_3D_Y, ANGLE_45)
QUAT_Y_90 = _axis_angle_to_quaternion(AXIS_3D_Y, ANGLE_90)
QUAT_Y_180 = _axis_angle_to_quaternion(AXIS_3D_Y, ANGLE_180)
QUAT_Z_45 = _axis_angle_to_quaternion(AXIS_3D_Z, ANGLE_45)
QUAT_Z_90 = _axis_angle_to_quaternion(AXIS_3D_Z, ANGLE_90)
QUAT_Z_180 = _axis_angle_to_quaternion(AXIS_3D_Z, ANGLE_180)


def _rotation_3d_x(angle):
  """Creates a 3d rotation matrix around the x axis.

  Args:
    angle: The angle.

  Returns:
    The 3d rotation matrix.
  """
  angle = angle.item()
  return np.array(((1., 0., 0.),
                   (0., np.cos(angle), -np.sin(angle)),
                   (0., np.sin(angle), np.cos(angle))))  # pyformat: disable


def _rotation_3d_y(angle):
  """Creates a 3d rotation matrix around the y axis.

  Args:
    angle: The angle.

  Returns:
    The 3d rotation matrix.
  """
  angle = angle.item()
  return np.array(((np.cos(angle), 0., np.sin(angle)),
                   (0., 1., 0.),
                   (-np.sin(angle), 0., np.cos(angle))))  # pyformat: disable


def _rotation_3d_z(angle):
  """Creates a 3d rotation matrix around the z axis.

  Args:
    angle: The angle.

  Returns:
    The 3d rotation matrix.
  """
  angle = angle.item()
  return np.array(((np.cos(angle), -np.sin(angle), 0.),
                   (np.sin(angle), np.cos(angle), 0.),
                   (0., 0., 1.)))  # pyformat: disable


MAT_3D_ID = np.eye(3)
MAT_3D_X_45 = _rotation_3d_x(ANGLE_45)
MAT_3D_X_90 = _rotation_3d_x(ANGLE_90)
MAT_3D_X_180 = _rotation_3d_x(ANGLE_180)
MAT_3D_Y_45 = _rotation_3d_y(ANGLE_45)
MAT_3D_Y_90 = _rotation_3d_y(ANGLE_90)
MAT_3D_Y_180 = _rotation_3d_y(ANGLE_180)
MAT_3D_Z_45 = _rotation_3d_z(ANGLE_45)
MAT_3D_Z_90 = _rotation_3d_z(ANGLE_90)
MAT_3D_Z_180 = _rotation_3d_z(ANGLE_180)
