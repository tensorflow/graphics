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
"""This module implements differentiable rasterization functionalities.

Rasterization is a widely used rendering technique to render images of 3D
scenes. The input is typically a scene, and the output consists of raster images
of the visible part of the scene. Each pixel in these images contains
information about appearance or depth. A typical implementation would iterate
over triangles by maintaining a depth buffer, which decides the visibility of
the triangles.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.rendering.camera import orthographic
from tensorflow_graphics.util import safe_ops
from tensorflow_graphics.util import shape


def get_barycentric_coordinates(triangle_vertices, pixels, name=None):
  """Computes the barycentric coordinates of pixels for 2D triangles.

  Barycentric coordinates of a point `p` are represented as coefficients
  $(w_1, w_2, w_3)$ corresponding to the masses placed at the vertices of a
  reference triangle if `p` is the center of mass. Barycentric coordinates are
  normalized so that $w_1 + w_2 + w_3 = 1$. These coordinates play an essential
  role in computing the pixel attributes (e.g. depth, color, normals, and
  texture coordinates) of a point lying on the surface of a triangle. The point
  `p` is inside the triangle if all of its barycentric coordinates are positive.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    triangle_vertices: A tensor of shape `[A1, ..., An, 3, 2]`, where the last
      two dimensions represents the `x` and `y` coordinates for each vertex of a
      2D triangle.
    pixels: A tensor of shape `[A1, ..., An, N, 2]`, where `N` represents the
      number of pixels, and the last dimension represents the `x` and `y`
      coordinates of each pixel.
    name: A name for this op that defaults to
      "rasterizer_get_barycentric_coordinates".

  Returns:
    barycentric_coordinates: A float tensor of shape `[A1, ..., An, N, 3]`,
      representing the barycentric coordinates.
    valid: A boolean tensor of shape `[A1, ..., An, N], which is `True` where
      pixels are inside the triangle, and `False` otherwise.
  """
  with tf.compat.v1.name_scope(name, "rasterizer_get_barycentric_coordinates",
                               [triangle_vertices, pixels]):
    triangle_vertices = tf.convert_to_tensor(value=triangle_vertices)
    pixels = tf.convert_to_tensor(value=pixels)

    shape.check_static(
        tensor=triangle_vertices,
        tensor_name="triangle_vertices",
        has_dim_equals=((-1, 2), (-2, 3)))
    shape.check_static(
        tensor=pixels, tensor_name="pixels", has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(triangle_vertices, pixels),
        last_axes=(-3, -3),
        broadcast_compatible=True)

    vertex_1, vertex_2, vertex_3 = tf.unstack(
        tf.expand_dims(triangle_vertices, axis=-3), axis=-2)
    vertex_x1, vertex_y1 = tf.unstack(vertex_1, axis=-1)
    vertex_x2, vertex_y2 = tf.unstack(vertex_2, axis=-1)
    vertex_x3, vertex_y3 = tf.unstack(vertex_3, axis=-1)
    pixels_x, pixels_y = tf.unstack(pixels, axis=-1)

    x1_minus_x3 = vertex_x1 - vertex_x3
    x3_minus_x2 = vertex_x3 - vertex_x2
    y3_minus_y1 = vertex_y3 - vertex_y1
    y2_minus_y3 = vertex_y2 - vertex_y3
    x_minus_x3 = pixels_x - vertex_x3
    y_minus_y3 = pixels_y - vertex_y3

    determinant = y2_minus_y3 * x1_minus_x3 - x3_minus_x2 * y3_minus_y1
    coordinate_1 = y2_minus_y3 * x_minus_x3 + x3_minus_x2 * y_minus_y3
    coordinate_1 = safe_ops.safe_signed_div(coordinate_1, determinant)
    coordinate_2 = y3_minus_y1 * x_minus_x3 + x1_minus_x3 * y_minus_y3
    coordinate_2 = safe_ops.safe_signed_div(coordinate_2, determinant)
    coordinate_3 = 1.0 - (coordinate_1 + coordinate_2)

    barycentric_coordinates = tf.stack(
        (coordinate_1, coordinate_2, coordinate_3), axis=-1)
    valid = tf.logical_and(
        tf.logical_and(coordinate_1 >= 0, coordinate_2 >= 0), coordinate_3 >= 0)
    return barycentric_coordinates, valid


def get_bounding_box(triangle_vertices, image_width, image_height, name=None):
  """Computes 2D bounding boxes for 2D triangles inside an image.

  This function rounds the estimated bounding box corners and therefore has zero
  derivative everywhere with respect to the vertices. Note that this does not
  prevent the rasterizer from being differentiable.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    triangle_vertices: A tensor of shape `[A1, ..., An, 3, 2], where the last
      two dimensions represent the `x` and `y` coordinates of each vertex in a
      triangle.
    image_width: A scalar tensor or a `float`.
    image_height: A scalar tensor or a `float`.
    name: A name for this op that defaults to "rasterizer_get_bounding_box".

  Returns:
    bottom_right_corner: A tensor of shape `[A1, ..., An, 2], where the last
      dimension represents the `x_max` and `y_max` of the 2D bounding boxes.
    top_left_corner: A tensor of shape `[A1, ..., An, 2], where the last
      dimension represents the `x_min` and `y_min` of the 2D bounding boxes.
  """
  with tf.compat.v1.name_scope(name, "rasterizer_get_bounding_box",
                               [triangle_vertices, image_width, image_height]):

    triangle_vertices = tf.convert_to_tensor(value=triangle_vertices)
    image_width = tf.convert_to_tensor(
        value=image_width, dtype=triangle_vertices.dtype)
    image_height = tf.convert_to_tensor(
        value=image_height, dtype=triangle_vertices.dtype)

    shape.check_static(
        tensor=triangle_vertices,
        tensor_name="triangle_vertices",
        has_dim_equals=((-1, 2), (-2, 3)))
    shape.check_static(
        tensor=image_width, tensor_name="image_width", has_rank=0)
    shape.check_static(
        tensor=image_height, tensor_name="image_height", has_rank=0)

    max_clip_values = tf.stack((image_width / 2, image_height / 2), axis=-1)
    min_clip_values = -max_clip_values
    clipped_vertices = tf.clip_by_value(triangle_vertices, min_clip_values,
                                        max_clip_values)
    bottom_right_corner = tf.reduce_max(input_tensor=clipped_vertices, axis=-2)
    bottom_right_corner = tf.math.ceil(bottom_right_corner)
    top_left_corner = tf.reduce_min(input_tensor=clipped_vertices, axis=-2)
    top_left_corner = tf.math.floor(top_left_corner)
    return bottom_right_corner, top_left_corner


def rasterize_triangle(index,
                       result_tensor,
                       vertices,
                       triangles,
                       project_function=orthographic.project,
                       min_depth=0.0,
                       name=None):
  """Rasterizes a single triangle.

  This implementation leverages the bounding boxes of the triangle to reduce
  the computational cost. Also, it assumes that the camera is at the origin with
  no rotation.

  Args:
    index: an integer, which represents the index of triangle.
    result_tensor: A tensor of shape `[H, W, 5]`, where H, W are the height and
      width of the rasterization output. For the last dimension, the first
      channel represents the rasterized depth map, the second represents the
      triangle index map, and the rest 3 channels represent the barycentric
      coordinate maps.
    vertices: A tensor of shape `[M, 3]`, where M is the number of vertices, the
      last dimension represents the x, y, z coordinates of each vertex.
    triangles: An integer tensor of shape `[N, 3]`, where N is the number of
      triangles in the mesh, and the last dimension represents the indices of
      each vertex in the triangle.
    project_function: A tensorflow function of a single tensor. This tensor of
      shape `[A1, ..., An, 3]`, where the last dimension represents a 3d point
      to project. It returns a tensor of shape `[A1, ..., An, 2]`, where the
      last dimension represents a 2d point.
    min_depth: A scalar tensor or a `float` that is used to determine the near
      plane depth to be used for clipping purposes.
    name: A name for this op that defaults to "rasterizer_rasterize_triangle".
  return: A tensor of shape `[H, W, 5]`, where H, W are the height and width of
    the rasterization output. This tensor holds the updated value of
    result_tensor.
  """
  with tf.compat.v1.name_scope(
      name, "rasterizer_rasterize_triangle",
      [index, result_tensor, vertices, triangles, min_depth]):
    index = tf.convert_to_tensor(value=index, dtype=tf.int32)
    result_tensor = tf.convert_to_tensor(value=result_tensor)
    vertices = tf.convert_to_tensor(value=vertices, dtype=result_tensor.dtype)
    triangles = tf.convert_to_tensor(value=triangles, dtype=tf.int32)
    min_depth = tf.convert_to_tensor(value=min_depth, dtype=result_tensor.dtype)

    shape.check_static(tensor=index, tensor_name="index", has_rank=0)
    shape.check_static(
        tensor=result_tensor,
        tensor_name="result_tensor",
        has_rank=3,
        has_dim_equals=(-1, 5))
    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_dim_equals=(-1, 3),
        has_rank=2)
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_dim_equals=(-1, 3),
        has_rank=2)

    vertices_indices = triangles[index]
    triangle_vertices_3d = tf.gather(vertices, vertices_indices, axis=-2)
    triangle_2d = project_function(triangle_vertices_3d)
    result_tensor_shape = tf.shape(input=result_tensor)
    image_width = tf.cast(result_tensor_shape[-2], result_tensor.dtype)
    image_height = tf.cast(result_tensor_shape[-3], result_tensor.dtype)
    bottom_right_corner, top_left_corner = get_bounding_box(
        triangle_2d, image_width, image_height)
    x, y = tf.meshgrid(
        tf.range(top_left_corner[0], bottom_right_corner[0]),
        tf.range(top_left_corner[1], bottom_right_corner[1]))
    pixels = tf.stack((tf.reshape(x, (-1,)), tf.reshape(y, (-1,))), axis=-1)
    indices = tf.stack(
        (pixels[:, 1] + image_height / 2.0, pixels[:, 0] + image_width / 2.0),
        axis=1)
    indices = tf.cast(indices, tf.int32)
    barycentric_coordinates, valid = get_barycentric_coordinates(
        triangle_2d, pixels)
    vertices_z = triangle_vertices_3d[..., 2]
    z = tf.reduce_sum(
        input_tensor=barycentric_coordinates * vertices_z, axis=-1)
    current_z = tf.gather_nd(result_tensor, indices)[..., 0]
    visible_max = tf.less(z, current_z)
    visible_min = tf.greater(z, min_depth)
    mask = tf.logical_and(tf.logical_and(visible_min, visible_max), valid)
    indices = tf.boolean_mask(tensor=indices, mask=mask)
    # Update depth_buffer, triangle_ind, and barycentric_coordinates.
    z = tf.expand_dims(tf.boolean_mask(tensor=z, mask=mask), axis=-1)
    triangle_index_update = tf.cast(index,
                                    result_tensor.dtype) * tf.ones_like(z)
    barycentric_coordinates = tf.boolean_mask(
        tensor=barycentric_coordinates, mask=mask)
    update = tf.concat((z, triangle_index_update, barycentric_coordinates),
                       axis=-1)
    return tf.compat.v1.tensor_scatter_nd_update(result_tensor, indices, update)


def rasterize_mesh(vertices,
                   triangles,
                   image_width,
                   image_height,
                   min_depth=0.0,
                   max_depth=float("Inf"),
                   output_dtype=tf.float32,
                   project_function=orthographic.project,
                   name=None):
  """Rasterize a single mesh into an image.

  This function rasterizes a single mesh into a result tensor concatenating
  a depth map, a triangle index map, and a barycentric coordinate map. The
  maximum value of depth map is set to be max_depth. For pixels that no triangle
  rasterize, their values in triangle index map and barycentric coordinate map
  will be -1.0.

  Args:
    vertices: A tensor of shape `[M, 3]`, where M is the number of vertices, the
      last dimension represents the x, y, z coordinates of each vertex.
    triangles: A tensor of shape `[N, 3]`, where N is the number of triangles in
      the mesh, and the last dimension represents the indices of each vertex in
      the triangle.
    image_width: An integer, which represents the width of the image.
    image_height: An integer, which represents the height of the image.
    min_depth: A scalar tensor or a `float` that is used to determine the near
      plane depth to be used for clipping purposes.
    max_depth: A float, which represents the maximum depth.
    output_dtype: A tf.DType specify the output dtype.
    project_function: a tensorflow function of a single tensor, points_3d.
      points_3d is a tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents a 3d point to project. It returns a tensor of shape
      `[A1, ..., An, 2]`, where the last dimension represents a 2d point.
    name: A name for this op that defaults to "rasterizer_rasterize_mesh".

  Returns:
    A tensor of shape `[image_height, image_width, 5]`, where for the last
      dimension, the first channel represents the rasterized depth map, the
      second second represents the triangle index map, and the third represents
      the barycentric coordinate maps.
  """
  with tf.compat.v1.name_scope(
      name, "rasterizer_rasterize_mesh",
      [vertices, triangles, image_width, image_height, min_depth, max_depth]):
    vertices = tf.convert_to_tensor(value=vertices, dtype=output_dtype)
    triangles = tf.convert_to_tensor(value=triangles, dtype=tf.int32)
    image_width = tf.convert_to_tensor(value=image_width, dtype=tf.int32)
    image_height = tf.convert_to_tensor(value=image_height, dtype=tf.int32)
    min_depth = tf.convert_to_tensor(value=min_depth, dtype=output_dtype)
    max_depth = tf.convert_to_tensor(value=max_depth, dtype=output_dtype)

    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_dim_equals=(-1, 3),
        has_rank=2)
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_dim_equals=(-1, 3),
        has_rank=2)
    shape.check_static(
        tensor=image_width, tensor_name="image_width", has_rank=0)
    shape.check_static(
        tensor=image_height, tensor_name="image_height", has_rank=0)
    shape.check_static(tensor=min_depth, tensor_name="min_depth", has_rank=0)
    shape.check_static(tensor=max_depth, tensor_name="max_depth", has_rank=0)

    num_triangles = tf.shape(input=triangles)[0]
    triangle_index = 0
    depth_image = max_depth * tf.ones((image_height, image_width, 1))
    initial_values = -tf.ones((image_height, image_width, 4))
    result_tensor = tf.concat((depth_image, initial_values), axis=-1)
    cond = lambda i, r: tf.less(i, num_triangles)

    def body(i, r):
      return (i + 1,
              rasterize_triangle(i, r, vertices, triangles, project_function,
                                 min_depth))

    updates = tf.while_loop(
        cond=cond, body=body, loop_vars=(triangle_index, result_tensor))
    with tf.control_dependencies(updates):
      return updates[1]


def rasterize(vertices,
              triangles,
              image_width,
              image_height,
              min_depth=0.0,
              max_depth=float("Inf"),
              output_dtype=tf.float32,
              project_function=orthographic.project,
              name=None):
  """Rasterize a batch of meshes.

  This function takes a batch of meshes and a projection function as input,
  and rasterizes the meshes into depth maps, triangle index maps, and
  barycentric coordinate maps. The camera is assumed to be at the origin,
  pointing towards +z direction, so the vertices must be transformed to the
  camera frame before rasterization. The image size is determined by
  `image_width` and `image_height`. The maximum value of depth map is set to be
  `max_depth`. The triangle index map and barycentric coordinate map will be -1
  for empty pixels. `min_depth` determines the depth of the near clipping plane.
  When optimizing with a loss function defined over a depth map, `min_depth` can
  be set to a negative value to prevent negative translation hypotheses from
  getting stuck.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices: A tensor of shape `[A1, ..., An, M, 3]`, where M is the number of
      vertices and the last dimension represents the x, y, z coordinates of each
      vertex.
    triangles: A tensor of shape `[A1, ..., An, N, 3]`, where N is the number of
      triangles and the last dimension represents the indices of each vertex in
      the triangle.
    image_width: A scalar tensor or a `float`, which represents the width of
      output images.
    image_height: A scalar tensor or a `float`, which represents the height of
      output images.
    min_depth: A scalar tensor or a `float` that is used to determine the near
      plane depth to be used for clipping purposes.
    max_depth: A scalar tensor or a `float`, which represents the maximum depth.
    output_dtype: A tf.DType that specifies the output `dtype` of the resulting
      depth maps and barycentric coordinates.
    project_function: a tensorflow function of a single tensor. This tensor is
      of shape `[A1, ..., An, 3]`, where the last dimension represents a 3d
      point to project. It returns a tensor of shape `[A1, ..., An, 2]`, where
      the last dimension represents a 2d point.
   name: A name for this op that defaults to "rasterizer_rasterize".

  Returns:
    A list of 3 tensors corresponding to:
    depth_maps: a tensor of shape `[A1, ..., An, image_height, image_width, 1]`,
      which represents the rasterized depth maps.
    triangle_ind_map: a tensor of shape `[A1, ..., An, image_height,
      image_width, 1]`, which represents the triangle indices.
    barycentric_coord_map: a tensor of shape `[A1, ..., An, image_height,
      image_width, 3]`, which represents the barycentric coordinates.

  """
  # TODO(b/137298417): Handle camera orientation and projective cameras
  with tf.compat.v1.name_scope(
      name, "rasterizer_rasterize",
      [vertices, triangles, image_width, image_height, min_depth, max_depth]):

    image_width = tf.convert_to_tensor(value=image_width, dtype=tf.int32)
    image_height = tf.convert_to_tensor(value=image_height, dtype=tf.int32)
    min_depth = tf.convert_to_tensor(value=min_depth, dtype=output_dtype)
    max_depth = tf.convert_to_tensor(value=max_depth, dtype=output_dtype)
    vertices = tf.convert_to_tensor(value=vertices, dtype=output_dtype)
    triangles = tf.convert_to_tensor(value=triangles, dtype=tf.int32)

    shape.check_static(
        tensor=image_width, tensor_name="image_width", has_rank=0)
    shape.check_static(
        tensor=image_height, tensor_name="image_height", has_rank=0)
    shape.check_static(tensor=min_depth, tensor_name="min_depth", has_rank=0)
    shape.check_static(tensor=max_depth, tensor_name="max_depth", has_rank=0)
    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(vertices, triangles),
        last_axes=(-3, -3),
        broadcast_compatible=False)

    if vertices.shape.ndims == 2:
      result_tensor = rasterize_mesh(vertices, triangles, image_width,
                                     image_height, min_depth, max_depth,
                                     output_dtype, project_function)
      return tf.compat.v1.split(result_tensor, (1, 1, 3), axis=-1)
    else:
      vertices_shape = vertices.shape.as_list()
      triangles_shape = triangles.shape.as_list()
      batch_shape = vertices_shape[:-2]
      vertices = tf.reshape(vertices, [-1] + vertices_shape[-2:])
      triangles = tf.reshape(triangles, [-1] + triangles_shape[-2:])
      batch_size = vertices.shape[0]
    batch_index = 0
    result = tf.zeros((batch_size, image_height, image_width, 5),
                      dtype=output_dtype)

    cond = lambda i, r: tf.less(i, batch_size)

    def update_result(i, r):
      update = [
          rasterize_mesh(vertices[i], triangles[i], image_width, image_height,
                         min_depth, max_depth, output_dtype, project_function)
      ]
      return tf.compat.v1.tensor_scatter_nd_update(r, ((i,),), update)

    body = lambda i, r: (i + 1, update_result(i, r))
    updated_results = tf.while_loop(
        cond=cond, body=body, loop_vars=(batch_index, result))

    with tf.control_dependencies(updated_results):
      new_result_tensor = updated_results[1]
      result_tensor_shape = new_result_tensor.shape.as_list()
      new_result_tensor = tf.reshape(new_result_tensor,
                                     batch_shape + result_tensor_shape[-3:])

    return tf.compat.v1.split(new_result_tensor, (1, 1, 3), axis=-1)
