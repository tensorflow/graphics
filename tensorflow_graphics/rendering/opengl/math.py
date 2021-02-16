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
"""This module implements math routines used by OpenGL."""

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import look_at
from tensorflow_graphics.math.interpolation import weighted
from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def model_to_eye(point_model_space,
                 camera_position,
                 look_at_point,
                 up_vector,
                 name=None):
  """Transforms points from model to eye coordinates.

  Note:
    In the following, A1 to An are optional batch dimensions which must be
    broadcast compatible.

  Args:
    point_model_space: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the 3D points in model space.
    camera_position: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the 3D position of the camera.
    look_at_point: A tensor of shape `[A1, ..., An, 3]`, with the last dimension
      storing the position where the camera is looking at.
    up_vector: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      defines the up vector of the camera.
    name: A name for this op. Defaults to 'model_to_eye'.

  Raises:
    ValueError: if the all the inputs are not of the same shape, or if any input
    of of an unsupported shape.

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, containing `point_model_space` in eye
    coordinates.
  """
  with tf.compat.v1.name_scope(
      name, "model_to_eye",
      [point_model_space, camera_position, look_at_point, up_vector]):
    point_model_space = tf.convert_to_tensor(value=point_model_space)
    camera_position = tf.convert_to_tensor(value=camera_position)
    look_at_point = tf.convert_to_tensor(value=look_at_point)
    up_vector = tf.convert_to_tensor(value=up_vector)

    shape.check_static(
        tensor=point_model_space,
        tensor_name="point_model_space",
        has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(point_model_space, camera_position),
        last_axes=-2,
        tensor_names=("point_model_space", "camera_position"),
        broadcast_compatible=True)

    model_to_eye_matrix = look_at.right_handed(camera_position, look_at_point,
                                               up_vector)
    batch_shape = tf.shape(input=point_model_space)[:-1]
    one = tf.ones(
        shape=tf.concat((batch_shape, (1,)), axis=-1),
        dtype=point_model_space.dtype)
    point_model_space = tf.concat((point_model_space, one), axis=-1)
    point_model_space = tf.expand_dims(point_model_space, axis=-1)
    res = tf.squeeze(tf.matmul(model_to_eye_matrix, point_model_space), axis=-1)
    return res[..., :-1]


def eye_to_clip(point_eye_space,
                vertical_field_of_view,
                aspect_ratio,
                near,
                far,
                name=None):
  """Transforms points from eye to clip space.

  Note:
    In the following, A1 to An are optional batch dimensions which must be
    broadcast compatible.

  Args:
    point_eye_space: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the 3D points in eye coordinates.
    vertical_field_of_view: A tensor of shape `[A1, ..., An, 1]`, where the last
      dimension represents the vertical field of view of the frustum. Note that
      values for `vertical_field_of_view` must be in the range ]0,pi[.
    aspect_ratio: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      stores the width over height ratio of the frustum. Note that values for
      `aspect_ratio` must be non-negative.
    near: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the near clipping plane. Note
      that values for `near` must be non-negative.
    far: A tensor of shape `[A1, ..., An, 1]`, where the last dimension captures
      the distance between the viewer and the far clipping plane. Note that
      values for `far` must be non-negative.
    name: A name for this op. Defaults to 'eye_to_clip'.

  Raises:
    ValueError: If any input is of an unsupported shape.

  Returns:
    A tensor of shape `[A1, ..., An, 4]`, containing `point_eye_space` in
    homogeneous clip coordinates.
  """
  with tf.compat.v1.name_scope(
      name, "eye_to_clip",
      [point_eye_space, vertical_field_of_view, aspect_ratio, near, far]):
    point_eye_space = tf.convert_to_tensor(value=point_eye_space)
    vertical_field_of_view = tf.convert_to_tensor(value=vertical_field_of_view)
    aspect_ratio = tf.convert_to_tensor(value=aspect_ratio)
    near = tf.convert_to_tensor(value=near)
    far = tf.convert_to_tensor(value=far)

    shape.check_static(
        tensor=point_eye_space,
        tensor_name="point_eye_space",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=vertical_field_of_view,
        tensor_name="vertical_field_of_view",
        has_dim_equals=(-1, 1))
    shape.check_static(
        tensor=aspect_ratio, tensor_name="aspect_ratio", has_dim_equals=(-1, 1))
    shape.check_static(tensor=near, tensor_name="near", has_dim_equals=(-1, 1))
    shape.check_static(tensor=far, tensor_name="far", has_dim_equals=(-1, 1))
    shape.compare_batch_dimensions(
        tensors=(point_eye_space, vertical_field_of_view, aspect_ratio, near,
                 far),
        last_axes=-2,
        tensor_names=("point_eye_space", "vertical_field_of_view",
                      "aspect_ratio", "near", "far"),
        broadcast_compatible=True)

    perspective_matrix = perspective.right_handed(vertical_field_of_view,
                                                  aspect_ratio, near, far)
    batch_shape = tf.shape(input=point_eye_space)[:-1]
    one = tf.ones(
        shape=tf.concat((batch_shape, (1,)), axis=-1),
        dtype=point_eye_space.dtype)
    point_eye_space = tf.concat((point_eye_space, one), axis=-1)
    point_eye_space = tf.expand_dims(point_eye_space, axis=-1)

    return tf.squeeze(tf.matmul(perspective_matrix, point_eye_space), axis=-1)


def clip_to_ndc(point_clip_space, name=None):
  """Transforms points from clip to normalized device coordinates (ndc).

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    point_clip_space: A tensor of shape `[A1, ..., An, 4]`, where the last
      dimension represents points in clip space.
    name: A name for this op. Defaults to 'clip_to_ndc'.

  Raises:
    ValueError: If `point_clip_space` is not of size 4 in its last dimension.

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, containing `point_clip_space` in
    normalized device coordinates.
  """
  with tf.compat.v1.name_scope(name, "clip_to_ndc", [point_clip_space]):
    point_clip_space = tf.convert_to_tensor(value=point_clip_space)

    shape.check_static(
        tensor=point_clip_space,
        tensor_name="point_clip_space",
        has_dim_equals=(-1, 4))

    w = point_clip_space[..., -1:]
    return point_clip_space[..., :3] / w


def ndc_to_screen(point_ndc_space,
                  lower_left_corner,
                  screen_dimensions,
                  near,
                  far,
                  name=None):
  """Transforms points from normalized device coordinates to screen coordinates.

  Note:
    In the following, A1 to An are optional batch dimensions which must be
    broadcast compatible between `point_ndc_space` and the other variables.

  Args:
    point_ndc_space: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents points in normalized device coordinates.
    lower_left_corner: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension captures the position (in pixels) of the lower left corner of
      the screen.
    screen_dimensions: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension is expressed in pixels and captures the width and the height (in
      pixels) of the screen.
    near:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the near clipping plane. Note
      that values for `near` must be non-negative.
    far:  A tensor of shape `[A1, ..., An, 1]`, where the last dimension
      captures the distance between the viewer and the far clipping plane. Note
      that values for `far` must be greater than those of `near`.
    name: A name for this op. Defaults to 'ndc_to_screen'.

  Raises:
    InvalidArgumentError: if any input contains data not in the specified range
      of valid values.
    ValueError: If any input is of an unsupported shape.

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, containing `point_ndc_space` in
    screen coordinates.
  """
  with tf.compat.v1.name_scope(
      name, "ndc_to_screen",
      [point_ndc_space, lower_left_corner, screen_dimensions, near, far]):
    point_ndc_space = tf.convert_to_tensor(value=point_ndc_space)
    lower_left_corner = tf.convert_to_tensor(value=lower_left_corner)
    screen_dimensions = tf.convert_to_tensor(value=screen_dimensions)
    near = tf.convert_to_tensor(value=near)
    far = tf.convert_to_tensor(value=far)

    shape.check_static(
        tensor=point_ndc_space,
        tensor_name="point_ndc_space",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=lower_left_corner,
        tensor_name="lower_left_corner",
        has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=screen_dimensions,
        tensor_name="screen_dimensions",
        has_dim_equals=(-1, 2))
    shape.check_static(tensor=near, tensor_name="near", has_dim_equals=(-1, 1))
    shape.check_static(tensor=far, tensor_name="far", has_dim_equals=(-1, 1))

    shape.compare_batch_dimensions(
        tensors=(lower_left_corner, screen_dimensions, near, far),
        last_axes=-2,
        tensor_names=("lower_left_corner", "screen_dimensions", "near", "far"),
        broadcast_compatible=False)
    shape.compare_batch_dimensions(
        tensors=(point_ndc_space, near),
        last_axes=-2,
        tensor_names=("point_ndc_space", "near"),
        broadcast_compatible=True)

    screen_dimensions = asserts.assert_all_above(
        screen_dimensions, 0.0, open_bound=True)
    near = asserts.assert_all_above(near, 0.0, open_bound=True)
    far = asserts.assert_all_above(far, near, open_bound=True)

    ndc_to_screen_factor = tf.concat(
        (screen_dimensions, far - near), axis=-1) / 2.0
    screen_center = tf.concat(
        (lower_left_corner + screen_dimensions / 2.0, (near + far) / 2.0),
        axis=-1)
    return ndc_to_screen_factor * point_ndc_space + screen_center


def model_to_screen(point_model_space,
                    model_to_eye_matrix,
                    perspective_matrix,
                    screen_dimensions,
                    lower_left_corner=(0.0, 0.0),
                    name=None):
  """Transforms points from model to screen coordinates.

  Note:
    Please refer to http://www.songho.ca/opengl/gl_transform.html for an
    in-depth review of this pipeline.

  Note:
    In the following, A1 to An are optional batch dimensions which must be
    broadcast compatible.

  Args:
    point_model_space: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the 3D points in model space.
    model_to_eye_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, where the last
      two dimension represent matrices to transform points from model to eye
      coordinates.
    perspective_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, where the last
      two dimension represent matrices to transform points from eye to clip
      coordinates.
    screen_dimensions: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension is expressed in pixels and captures the width and the height (in
      pixels) of the screen.
    lower_left_corner: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension captures the position (in pixels) of the lower left corner of
      the screen.
    name: A name for this op. Defaults to 'model_to_screen'.

  Raises:
    InvalidArgumentError: if any input contains data not in the specified range
      of valid values.
    ValueError: If any input is of an unsupported shape.

  Returns:
    A tuple of two tensors, respectively of shape `[A1, ..., An, 3]` and
    `[A1, ..., An, 1]`, where the first tensor containing the projection of
    `point_model_space` in screen coordinates, and the second represents the 'w'
    component of `point_model_space` in clip space.
  """
  with tf.compat.v1.name_scope(name, "model_to_screen", [
      point_model_space, model_to_eye_matrix, perspective_matrix,
      screen_dimensions, lower_left_corner
  ]):
    point_model_space = tf.convert_to_tensor(value=point_model_space)
    model_to_eye_matrix = tf.convert_to_tensor(value=model_to_eye_matrix)
    perspective_matrix = tf.convert_to_tensor(value=perspective_matrix)

    shape.check_static(
        tensor=point_model_space,
        tensor_name="point_model_space",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=model_to_eye_matrix,
        tensor_name="model_to_eye_matrix",
        has_dim_equals=((-1, 4), (-2, 4)))
    shape.check_static(
        tensor=perspective_matrix,
        tensor_name="perspective_matrix",
        has_dim_equals=((-1, 4), (-2, 4)))
    shape.compare_batch_dimensions(
        tensors=(point_model_space, model_to_eye_matrix, perspective_matrix),
        last_axes=(-2, -3, -3),
        tensor_names=("point_model_space", "model_to_eye_matrix",
                      "perspective_matrix"),
        broadcast_compatible=True)

    batch_shape = tf.shape(input=point_model_space)[:-1]
    one = tf.ones(
        shape=tf.concat((batch_shape, (1,)), axis=-1),
        dtype=point_model_space.dtype)
    point_model_space = tf.concat((point_model_space, one), axis=-1)
    point_model_space = tf.expand_dims(point_model_space, axis=-1)

    view_projection_matrix = tf.linalg.matmul(perspective_matrix,
                                              model_to_eye_matrix)

    _, _, near, far = perspective.parameters_from_right_handed(
        perspective_matrix)

    point_clip_space = tf.squeeze(
        tf.matmul(view_projection_matrix, point_model_space), axis=-1)
    point_ndc_space = clip_to_ndc(point_clip_space)
    point_screen_space = ndc_to_screen(point_ndc_space, lower_left_corner,
                                       screen_dimensions, near, far)
    return point_screen_space, point_clip_space[..., 3:4]


def perspective_correct_barycentrics(triangle_vertices_model_space,
                                     pixel_position,
                                     model_to_eye_matrix,
                                     perspective_matrix,
                                     screen_dimensions,
                                     lower_left_corner=(0.0, 0.0),
                                     name=None):
  """Computes perspective correct barycentrics.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    triangle_vertices_model_space: A tensor of shape `[A1, ..., An, 3, 3]`,
      where the last dimension represents the vertices of a triangle in model
      space.
    pixel_position: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension stores the position (in pixels) where the interpolation is
      requested.
    model_to_eye_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, where the last
      two dimension represent matrices to transform points from model to eye
      coordinates.
    perspective_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, where the last
      two dimension represent matrices to transform points from eye to clip
      coordinates.
    screen_dimensions: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension is expressed in pixels and captures the width and the height (in
      pixels) of the screen.
    lower_left_corner: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension captures the position (in pixels) of the lower left corner of
      the screen.
    name: A name for this op. Defaults to 'perspective_correct_barycentrics'.

  Raises:
    InvalidArgumentError: if any input contains data not in the specified range
      of valid values.
    ValueError: If any input is of an unsupported shape.

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, containing perspective correct
    barycentric coordinates.
  """
  with tf.compat.v1.name_scope(name, "perspective_correct_barycentrics", [
      triangle_vertices_model_space, pixel_position, model_to_eye_matrix,
      perspective_matrix, screen_dimensions, lower_left_corner
  ]):
    pixel_position = tf.convert_to_tensor(value=pixel_position)
    triangle_vertices_model_space = tf.convert_to_tensor(
        value=triangle_vertices_model_space)
    shape.check_static(
        tensor=pixel_position,
        tensor_name="pixel_position",
        has_dim_equals=(-1, 2))
    shape.check_static(
        tensor=triangle_vertices_model_space,
        tensor_name="triangle_vertices_model_space",
        has_dim_equals=((-2, 3), (-1, 3)))

    lower_left_corner = tf.convert_to_tensor(value=lower_left_corner)
    screen_dimensions = tf.convert_to_tensor(value=screen_dimensions)
    lower_left_corner = shape.add_batch_dimensions(
        lower_left_corner,
        "lower_left_corner",
        model_to_eye_matrix.shape[:-2],
        last_axis=-2)
    screen_dimensions = shape.add_batch_dimensions(
        screen_dimensions,
        "screen_dimensions",
        model_to_eye_matrix.shape[:-2],
        last_axis=-2)

    vertices_screen, vertices_w = model_to_screen(triangle_vertices_model_space,
                                                  model_to_eye_matrix,
                                                  perspective_matrix,
                                                  screen_dimensions,
                                                  lower_left_corner)
    vertices_w = tf.squeeze(vertices_w, axis=-1)
    pixel_position = tf.expand_dims(pixel_position, axis=-2)
    barycentric_coordinates, _ = weighted.get_barycentric_coordinates(
        vertices_screen[..., :2], pixel_position)
    barycentric_coordinates = tf.squeeze(barycentric_coordinates, axis=-2)
    coeffs = barycentric_coordinates / vertices_w
    return tf.linalg.normalize(coeffs, ord=1, axis=-1)[0]


def interpolate_attributes(attribute, barycentric, name=None):
  """Interpolates attributes using barycentric weights.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    attribute: A tensor of shape `[A1, ..., An, 3, B]`, where the last dimension
      stores a per-vertex `B`-dimensional attribute.
    barycentric: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      contains barycentric coordinates.
    name: A name for this op. Defaults to 'interpolate_attributes'.

  Returns:
    A tensor of shape `[A1, ..., An, B]`, containing interpolated attributes.
  """
  with tf.compat.v1.name_scope(name, "interpolate_attributes",
                               (attribute, barycentric)):
    attribute = tf.convert_to_tensor(value=attribute)
    barycentric = tf.convert_to_tensor(value=barycentric)

    shape.check_static(
        tensor=attribute, tensor_name="attribute", has_dim_equals=(-2, 3))
    shape.check_static(
        tensor=barycentric, tensor_name="barycentric", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(attribute, barycentric),
        last_axes=(-2, -1),
        tensor_names=("attribute", "barycentric"),
        broadcast_compatible=True)
    barycentric = asserts.assert_normalized(barycentric, order=1)
    return tf.reduce_sum(
        input_tensor=tf.expand_dims(barycentric, axis=-1) * attribute, axis=-2)


def perspective_correct_interpolation(triangle_vertices_model_space,
                                      attribute,
                                      pixel_position,
                                      model_to_eye_matrix,
                                      perspective_matrix,
                                      screen_dimensions,
                                      lower_left_corner=(0.0, 0.0),
                                      name=None):
  """Returns perspective corrected interpolation of attributes over triangles.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    triangle_vertices_model_space: A tensor of shape `[A1, ..., An, 3, 3]`,
      where the last dimension represents the vertices of a triangle in model
      space.
    attribute: A tensor of shape `[A1, ..., An, 3, B]`, where the last dimension
      stores a per-vertex `B`-dimensional attribute.
    pixel_position: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension stores the position (in pixels) where the interpolation is
      requested.
    model_to_eye_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, where the last
      two dimension represent matrices to transform points from model to eye
      coordinates.
    perspective_matrix: A tensor of shape `[A1, ..., An, 4, 4]`, where the last
      two dimension represent matrices to transform points from eye to clip
      coordinates.
    screen_dimensions: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension is expressed in pixels and captures the width and the height (in
      pixels) of the screen.
    lower_left_corner: A tensor of shape `[A1, ..., An, 2]`, where the last
      dimension captures the position (in pixels) of the lower left corner of
      the screen.
    name: A name for this op. Defaults to 'perspective_correct_interpolation'.

  Raises:
    tf.errors.InvalidArgumentError: if any input contains data not in the
      specified range of valid values.
    ValueError: If any input is of an unsupported shape.

  Returns:
    A tensor of shape `[A1, ..., An, B]`, containing interpolated attributes.
  """
  with tf.compat.v1.name_scope(name, "perspective_correct_interpolation", [
      triangle_vertices_model_space, attribute, pixel_position,
      model_to_eye_matrix, perspective_matrix, screen_dimensions,
      lower_left_corner
  ]):
    barycentric = perspective_correct_barycentrics(
        triangle_vertices_model_space, pixel_position, model_to_eye_matrix,
        perspective_matrix, screen_dimensions, lower_left_corner)
    return interpolate_attributes(attribute, barycentric)

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
