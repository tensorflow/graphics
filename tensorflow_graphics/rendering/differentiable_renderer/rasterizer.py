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
"""This module implements rasterization functionalites.

Rasterization is a widely used rendering technique to render
images of 3D scenes. which takes a scene described by geometric primitives as
input, and returns raster images of the visible part of the scene. Each point in
these images usually contains information about appearance or depth. A
rasterizer iterates over triangles by maintaining a depth buffer, which decides
the visibility of triangles.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.rendering.camera import orthographic
from tensorflow_graphics.util import shape


def _edge_function(edge_point_1, edge_point_2, point_2d):
  """Finds which side of the edge, defined by edge_point_1 and 2, point_2d is.

  This function has been presented by Juan Pineda in a 1988 paper called "A
  Parallel Algorithm for Polygon Rasterization". When testing on which side of
  the edge, defined by connecting 2D points from edge_point_1 to edge_point_2,
  point_2d is, the function returns a negative number when it is to the left of
  the edge, a positive number when it is to the right of this line, and zero,
  when the point is exactly on the line. When this function is positive, its
  value equals to 2 times the area formed by the three points.

  Note:
    In the following, A1 to An are optional batch dimensions. Broadcasting is
    supported.
  Args:
    edge_point_1: a tensors of shape `[A1, ..., An, 2]`. It represents the
      starting point of the edge. Its last dimension represents its x, y
      coordinates.
    edge_point_2: a tensors of shape `[A1, ..., An, 2]`. It represents the end
      point of the edge. Its last dimension represents its x, y coordinates.
    point_2d: a tensors of shape `[A1, ..., An, 2]`. It represents the 2D points
      to test. Its last dimension represents the x, y coordinates.

  Returns:
    A tensor of shape `[A1, ..., An]`
  """
  p1x = edge_point_1[..., 0]
  p1y = edge_point_1[..., 1]
  p2x = edge_point_2[..., 0]
  p2y = edge_point_2[..., 1]
  p3x = point_2d[..., 0]
  p3y = point_2d[..., 1]
  return (p2y - p1y) * (p3x - p1x) - (p2x - p1x) * (p3y - p1y)


def rasterizer_barycentric_coordinates(triangle_vertices_2d, triangle_vertices_attributes, pixels, name=None):
  """Compute the barycentric coordinates of pixels for 2D triangle vertices.

  Barycentric coordinates of a point p represented as coefficients
  $(w_0, w_1, w_2)$ corresponding to the masses placed at the vertices of a
  reference triangle if p is the mass centroid. Barycentric coordinates are
  normalized so that $$w_0 + w_1 + w_2 = 1$$. These barycentric coordinates play
  an essential role in computing the correct pixel attributes (e.g. depth,
  color, normal, and texture coordinates) of a point lying on the surface of a
  triangle. Knowing the vertex attributes $(a0, a1, a2)$ of each vertex as well
  as the barycentric coordinates allows to interpolate attribute values at any
  point. Also, when and only when $w_0 >0, w_1>0, w_2>0$, the point p is
  located inside the triangle.


  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    triangle_vertices_2d: A tensor of shape `[A1, ..., An, 3, 2], where the last
      two dimensions represents the x, y coordinate for each vertex of a 2D
      triangle.
    triangle_vertices_attributes: A tensor of shape `[A1, ..., An, 3, 2], where the last three dimensions represents the vertex attributes 
    pixels: A tensor of shape `[A1, ..., An, N, 2]`,where N represents the
      number of pixels, and the last dimension represents the x, y coordinates
      of each pixel.
    name: A name for this op that defaults to
      "rasterizer_barycentric_coordinates".

  Returns:
    barycentric_coordinates: A float tensor of shape `[A1, ..., An, N, 3]`,
      where N is the number of pixels in pixels. This tensor represents the
      barycentric coordinates.
    interpolated_attributes: A float tensor of shape `[A1, ..., An, N, 3]`,
    where N is the number of pixels in pixels. This tensor represents the interpolated vertex attributes.
    valid: A boolean tensor of shape `[A1, ..., An, N], where N is the number
     of pixels. This tensor is `True` where pixel is inside the triangle, and
     `False` otherwise.
  """
  with tf.compat.v1.name_scope(name, "rasterizer_barycentric_coordinates",
                               [triangle_vertices_2d, pixels]):
    triangle_vertices_2d = tf.convert_to_tensor(value=triangle_vertices_2d)
    pixels = tf.convert_to_tensor(value=pixels)

    shape.check_static(
        tensor=triangle_vertices_2d,
        tensor_name="triangle_vertices_2d",
        has_dim_equals=((-1, 2), (-2, 3)))
    shape.check_static(
        tensor=pixels, tensor_name="pixels", has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(triangle_vertices_2d, pixels),
        last_axes=(-3, -3),
        broadcast_compatible=True)

    vertex_0, vertex_1, vertex_2 = tf.unstack(
        tf.expand_dims(triangle_vertices_2d, axis=-3), axis=-2)

    # In the following, area is the 2 * the actual area of the triangle defined
    # by vertex_0, vertex_1, and vertex_2. The same applies to w0, w1, w2.
    area = _edge_function(vertex_0, vertex_1, vertex_2)
    w0 = _edge_function(vertex_1, vertex_2, pixels)
    w1 = _edge_function(vertex_2, vertex_0, pixels)
    w2 = _edge_function(vertex_0, vertex_1, pixels)

    barycentric_coordinates = tf.stack([w0, w1, w2], axis=-1) / tf.expand_dims(
        area, axis=-1)
    valid = tf.logical_and(tf.logical_and(w0 >= 0, w1 >= 0), w2 >= 0)
    interpolated_attributes = tf.matmul(barycentric_coordinates, triangle_vertices_attributes)

    return barycentric_coordinates, interpolated_attributes, valid


def rasterizer_bounding_box(triangle_vertices_2d,
                            image_width,
                            image_height,
                            name=None):
  """Compute 2D bounding boxes for 2D triangles inside an image.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    triangle_vertices_2d: A tensor of shape `[A1, ..., An, 3, 2], where the last
      two dimensions represent the x, y coordinates for each vertex in a
      triangle.
    image_width: A scalar tensor or an `int`.
    image_height: A scalar tensor or an `int`.
    name: A name for this op that defaults to "rasterizer_bounding_box".

  Returns:
    bottom_right_corner: A tensor of shape `[A1, ..., An, 2], where the last
      dimension represents the x_max, y_max of the 2D bounding boxes.
    top_left_corner: A tensor of shape `[A1, ..., An, 2], where the last
      dimension represents the x_min, y_min of the 2D bounding boxes.
  """
  with tf.compat.v1.name_scope(
      name, "rasterizer_bounding_box",
      [triangle_vertices_2d, image_width, image_height]):

    triangle_vertices_2d = tf.convert_to_tensor(value=triangle_vertices_2d)
    image_width = tf.convert_to_tensor(
        value=image_width, dtype=triangle_vertices_2d.dtype)
    image_height = tf.convert_to_tensor(
        value=image_height, dtype=triangle_vertices_2d.dtype)

    shape.check_static(
        tensor=triangle_vertices_2d,
        tensor_name="triangle_vertices_2d",
        has_dim_equals=((-1, 2), (-2, 3)))
    shape.check_static(
        tensor=image_width, tensor_name="image_width", has_rank=0)
    shape.check_static(
        tensor=image_height, tensor_name="image_height", has_rank=0)

    # clipping by the image size
    x, y = tf.unstack(triangle_vertices_2d, axis=-1)
    x = tf.clip_by_value(x, -image_width / 2, image_width / 2 - 1)
    y = tf.clip_by_value(y, -image_height / 2, image_height / 2 - 1)
    xy = tf.stack((x, y), axis=-1)

    # calculate the bottom right and top left corners of the 2d bounding box
    bottom_right_corner = tf.reduce_max(input_tensor=xy, axis=-2)
    bottom_right_corner = tf.math.ceil(bottom_right_corner)
    top_left_corner = tf.reduce_min(input_tensor=xy, axis=-2)
    top_left_corner = tf.math.floor(top_left_corner)
    return bottom_right_corner, top_left_corner


def rasterizer_rasterize_triangle(index,
                                  result_tensor,
                                  vertices,
                                  attributes,
                                  triangles,
                                  project_function=orthographic.project,
                                  name=None):
  """Rasterize a single triangle and then update the result tensor.

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
    attributes: A tensor of shape `[M, 3]`, where M is the number of vertices, the last dimension represents the attributes of each vertex.
    triangles: An integer tensor of shape `[N, 3]`, where N is the number of
      triangles in the mesh, and the last dimension represents the indices of
      each vertex in the triangle.
    project_function: a tensorflow function of a single tensor. This tensor of
      shape `[A1, ..., An, 3]`, where the last dimension represents a 3d point
      to project. It returns a tensor of shape `[A1, ..., An, 2]`, where the
      last dimension represents a 2d point.
    name: A name for this op that defaults to "rasterizer_rasterize_triangle".
  return: A tensor of shape `[H, W, 5]`, where H, W are the height and width of
    the rasterization output. This tensor holds the updated value of
    result_tensor.
  """
  with tf.compat.v1.name_scope(name, "rasterizer_rasterize_triangle",
                               [index, result_tensor, vertices, triangles]):
    index = tf.convert_to_tensor(value=index, dtype=tf.int32)
    result_tensor = tf.convert_to_tensor(value=result_tensor)
    vertices = tf.convert_to_tensor(value=vertices, dtype=result_tensor.dtype)
    attributes = tf.convert_to_tensor(value=attributes, dtype=tf.float32)
    triangles = tf.convert_to_tensor(value=triangles, dtype=tf.int32)

    shape.check_static(tensor=index, tensor_name="index", has_rank=0)
    shape.check_static(
        tensor=result_tensor,
        tensor_name="result_tensor",
        has_rank=3,
        has_dim_equals=(-1, 8))
    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_dim_equals=(-1, 3),
        has_rank=2)
    shape.check_static(
        tensor=attributes,
        tensor_name="attributes",
        has_dim_equals=(-1, 3),
        has_rank=2)
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_dim_equals=(-1, 3),
        has_rank=2)

    # Select triangle.
    vertices_indices = triangles[index]
    triangle_vertices_3d = tf.gather(vertices, vertices_indices, axis=-2)
    attributes_vertices_3d = tf.gather(attributes, vertices_indices, axis=-2)

    # Project triangle vertices onto 2D image plane.
    triangle_2d = project_function(triangle_vertices_3d)

    # Compute a 2D bounding box for the triangle.
    image_width = tf.cast(result_tensor.shape[-2], result_tensor.dtype)
    image_height = tf.cast(result_tensor.shape[-3], result_tensor.dtype)
    bottom_right_corner, top_left_corner = rasterizer_bounding_box(
        triangle_2d, image_width, image_height)

    # Compute the coordinates of the pixels inside the bounding box.
    x, y = tf.meshgrid(
        tf.range(top_left_corner[0], bottom_right_corner[0]),
        tf.range(top_left_corner[1], bottom_right_corner[1]))
    pixels = tf.stack([tf.reshape(x, (-1,)), tf.reshape(y, (-1,))], axis=-1)
    indices = tf.stack(
        [pixels[:, 1] + image_height / 2.0, pixels[:, 0] + image_width / 2.0],
        axis=1)
    indices = tf.cast(indices, tf.int32)

    # Compute the barycentric coordinates.
    barycentric_coordinates, interpolated_attributes, valid = rasterizer_barycentric_coordinates(
        triangle_2d, attributes_vertices_3d, pixels)

    # Compute updated mask according to depth buffer.
    vertices_z = triangle_vertices_3d[..., 2]
    z = tf.reduce_sum(
        input_tensor=barycentric_coordinates * vertices_z, axis=-1)
    current_z = tf.gather_nd(result_tensor, indices)[..., 0]
    visible = tf.less(z, current_z)
    mask = tf.logical_and(valid, visible)
    indices = tf.boolean_mask(tensor=indices, mask=mask)

    # Update depth_buffer, triangle_ind, and barycentric_coordinates.
    z = tf.expand_dims(tf.boolean_mask(tensor=z, mask=mask), axis=-1)
    triangle_index_update = tf.cast(index,
                                    result_tensor.dtype) * tf.ones_like(z)
    barycentric_coordinates = tf.boolean_mask(
        tensor=barycentric_coordinates, mask=mask)
    interpolated_attributes = tf.boolean_mask(
        tensor=interpolated_attributes, mask=mask)

    update = tf.concat([z, triangle_index_update, barycentric_coordinates, interpolated_attributes],
                       axis=-1)

    return tf.compat.v1.scatter_nd_update(
        tf.Variable(result_tensor), indices, update)


def _rasterize_mesh(vertices,
                    attributes,
                    triangles,
                    image_width,
                    image_height,
                    max_depth=float("Inf"),
                    output_dtype=tf.float32,
                    project_function=orthographic.project):
  """Rasterize a single mesh into an image.

  This function rasterizes a single mesh into a result tensor concatenating
  a depth map, a triangle index map, and a barycentric coordinate map. The
  maximum value of depth map is set to be max_depth. For pixels that no triangle
  rasterize, their values in triangle index map and barycentric coordinate map
  will be -1.0.

  Args:
    vertices: A tensor of shape `[M, 3]`, where M is the number of vertices, the
      last dimension represents the x, y, z coordinates of each vertex.
    attributes: A tensor of shape `[M, 3]`, where M is the number of vertices, the last dimension represents the attributes of each vertex.
    triangles: A tensor of shape `[N, 3]`, where N is the number of triangles in
      the mesh, and the last dimension represents the indices of each vertex in
      the triangle.
    image_width: An integer, which represents the width of the image.
    image_height: An integer, which represents the height of the image.
    max_depth: A float, which represents the maximum depth.
    output_dtype: A tf.DType specify the output dtype.
    project_function: a tensorflow function of a single tensor, points_3d.
      points_3d is a tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents a 3d point to project. It returns a tensor of shape
      `[A1, ..., An, 2]`, where the last dimension represents a 2d point.

  Returns:
    A tensor of shape `[image_height, image_width, 5]`, where for the last
      dimension, the first channel represents the rasterized depth map, the
      second second represents the triangle index map, the third represents
      the barycentric coordinate maps and the fourth represents the interpolated attributes map
  """
  vertices = tf.convert_to_tensor(value=vertices, dtype=output_dtype)
  attributes = tf.convert_to_tensor(value=attributes, dtype=tf.float32)
  triangles = tf.convert_to_tensor(value=triangles, dtype=tf.int32)
  image_width = tf.convert_to_tensor(value=image_width, dtype=tf.int32)
  image_height = tf.convert_to_tensor(value=image_height, dtype=tf.int32)
  max_depth = tf.convert_to_tensor(value=max_depth, dtype=output_dtype)

  shape.check_static(
      tensor=vertices,
      tensor_name="vertices",
      has_dim_equals=(-1, 3),
      has_rank=2)
  shape.check_static(
      tensor=attributes,
      tensor_name="attributes",
      has_dim_equals=(-1, 3),
      has_rank=2)
  shape.check_static(
      tensor=triangles,
      tensor_name="triangles",
      has_dim_equals=(-1, 3),
      has_rank=2)
  shape.check_static(tensor=image_width, tensor_name="image_width", has_rank=0)
  shape.check_static(
      tensor=image_height, tensor_name="image_height", has_rank=0)
  shape.check_static(tensor=max_depth, tensor_name="max_depth", has_rank=0)

  num_triangles = triangles.shape[0]
  triangle_index = 0
  # Store depth_buffer, triangle_index, and barycentric_coordinates
  # in result_tensor.
  result_tensor = tf.concat([
      max_depth * tf.ones((image_height, image_width, 1)),
      -1.0 * tf.ones((image_height, image_width, 1)),
      -1.0 * tf.ones((image_height, image_width, 3)),
      -1.0 * tf.ones((image_height, image_width, 3))
  ], axis=-1)

  cond = lambda i, r: tf.less(i, num_triangles)

  def body(i, r):
    return (i + 1,
            rasterizer_rasterize_triangle(i, r, vertices, attributes, triangles,
                                          project_function))

  updates = tf.while_loop(
      cond=cond, body=body, loop_vars=(triangle_index, result_tensor))
  with tf.control_dependencies(updates):
    return updates[1]


def rasterizer_rasterize(vertices,
                         attributes,
                         triangles,
                         image_width,
                         image_height,
                         max_depth=float("Inf"),
                         output_dtype=tf.float32,
                         project_function=orthographic.project,
                         name=None):
  """Rasterize a batch of meshes.

  This function takes a batch of meshes and a projection function as input,
  rasterize the meshes into depth maps, triangle index maps, and barycentric
  coordinate maps with size of image_width and image_height. The maximum value
  of depth map is set to be max_depth. For pixels that no triangle rasterizes,
  their values in triangle index map and barycentric coordinate map will be -1.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices: A tensor of shape `[A1, ..., An, M, 3]` or a list of tensors of
      shape `[M, 3]` with length B, where B is the batch size, M is the number
      of vertices, and the last dimension of the tensors represent the x, y, z
      coordinates of each vertex.
    attributes: A tensor of shape `[A1, ..., An, M, 3]` or a list of tensors of
      shape `[M, 3]` with length B, where B is the batch size, M is the number
      of vertices, and the last dimension of the tensors represent the attributes of each vertex.
    triangles: A tensor of shape `[A1, ..., An, N, 3]` or a list of tensors of
      shape `[N, 3]` with length B, where B is the batch size, N is the number
      of triangles, and the last dimension of the tensors represents the indices
      of each vertex in the triangle.
    image_width: A scalar tensor, which represents the width of output images.
    image_height: A scalar tensor, which represents the height of output images.
    max_depth: A scalar tensor, which represents the maximum depth.
    output_dtype: A tf.DType specify the output dtype of depth_maps and
      barycentric_coord_map.
    project_function: a tensorflow function of a single tensor. This tensor is
      of shape `[A1, ..., An, 3]`, where the last dimension represents a 3d
      point to project. It returns a tensor of shape `[A1, ..., An, 2]`, where
      the last dimension represents a 2d point.
   name: A name for this op that defaults to "rasterizer_rasterize".

  Returns:
    depth_maps: a tensor of shape `[A1, ..., An, image_height, image_weith]`,
      which represents the rasterized depth maps.
    triangle_ind_map: an int32 tensor of shape
      `[A1, ..., An, image_height, image_width]`, which represents the triangle
      index maps.
    barycentric_coord_map: a tensor of shape
      `[A1, ..., An, image_height, image_weith,3]`, which represents the
      barycentric coordinate maps.
    interpolated_attributes: a tensor of shape `[A1, ..., An, image_height, image_weith,3]`, which represents the interpolated attributes map

  """
  with tf.compat.v1.name_scope(
      name, "rasterizer_rasterize",
      [vertices, attributes, triangles, image_width, image_height, max_depth]):

    image_width = tf.convert_to_tensor(value=image_width, dtype=tf.int32)
    image_height = tf.convert_to_tensor(value=image_height, dtype=tf.int32)
    max_depth = tf.convert_to_tensor(value=max_depth, dtype=output_dtype)

    shape.check_static(
        tensor=image_width, tensor_name="image_width", has_rank=0)
    shape.check_static(
        tensor=image_height, tensor_name="image_height", has_rank=0)
    shape.check_static(tensor=max_depth, tensor_name="max_depth", has_rank=0)

    list_flag = isinstance(vertices, list)
    if list_flag:
      batch_size = len(vertices)
    else:
      vertices = tf.convert_to_tensor(value=vertices, dtype=output_dtype)
      attributes = tf.convert_to_tensor(value=attributes, dtype=tf.float32)
      triangles = tf.convert_to_tensor(value=triangles, dtype=tf.int32)

      shape.check_static(
          tensor=vertices,
          tensor_name="vertices",
          has_rank_greater_than=1,
          has_dim_equals=(-1, 3))
      shape.check_static(
          tensor=attributes,
          tensor_name="attributes",
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
        result_tensor = _rasterize_mesh(vertices, attributes, triangles, image_width,
                                        image_height, max_depth, output_dtype,
                                        project_function)
        depth_maps = result_tensor[..., 0]
        triangle_index_map = tf.cast(result_tensor[..., 1], dtype=tf.int32)
        barycentric_coordinates = result_tensor[..., 2:5]
        interpolated_attributes = result_tensor[..., 5:]
        return depth_maps, triangle_index_map, barycentric_coordinates, interpolated_attributes
      else:
        vertices_shape = vertices.shape.as_list()
        attributes_shape = attributes.shape.as_list()
        triangles_shape = triangles.shape.as_list()
        batch_shape = vertices_shape[:-2]
        vertices = tf.reshape(vertices, [-1] + vertices_shape[-2:])
        attributes = tf.reshape(attributes, [-1] + attributes_shape[-2:])
        triangles = tf.reshape(triangles, [-1] + triangles_shape[-2:])
        batch_size = vertices.shape[0]

    batch_index = 0
    result = tf.zeros([batch_size, image_height, image_width, 8],
                      dtype=output_dtype)

    cond = lambda i, r: tf.less(i, batch_size)

    def update_result(i, r):
      update = [
          _rasterize_mesh(vertices[i], attributes[i], triangles[i], image_width, image_height,
                          max_depth, output_dtype, project_function)
      ]
      return tf.compat.v1.scatter_nd_update(tf.Variable(r), [[i]], update)

    body = lambda i, r: (i + 1, update_result(i, r))
    updated_results = tf.while_loop(
        cond=cond, body=body, loop_vars=(batch_index, result))

    with tf.control_dependencies(updated_results):
      new_result_tensor = updated_results[1]
      if not list_flag:
        result_tensor_shape = new_result_tensor.shape.as_list()
        new_result_tensor = tf.reshape(new_result_tensor,
                                       batch_shape + result_tensor_shape[-3:])

      depth_maps = new_result_tensor[..., 0]
      triangle_index_map = tf.cast(new_result_tensor[..., 1], dtype=tf.int32)
      barycentric_coordinates = new_result_tensor[..., 2:5]
      interpolated_attributes = new_result_tensor[..., 5:]

    return depth_maps, triangle_index_map, barycentric_coordinates, interpolated_attributes
