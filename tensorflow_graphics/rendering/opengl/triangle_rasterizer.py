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
"""This module implements a differentiable rasterizer of triangular meshes.

The resulting rendering contains perspective-correct interpolation of attributes
defined at the vertices of the rasterized meshes. This rasterizer does not
provide gradients through visibility, but it does through visible geometry and
attributes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_graphics.rendering.opengl import gen_rasterizer_op as render_ops
from tensorflow_graphics.rendering.opengl import math as glm
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _dim_value(dim):
  return 1 if dim is None else tf.compat.v1.dimension_value(dim)


# TODO(b/149683925): Put the shaders in separate files for reusability &
# code cleanliness.

# Empty vertex shader; all the work happens in the geometry shader.
vertex_shader = """
#version 430
void main() { }
"""

# Geometry shader that projects the vertices of visible triangles onto the image
# plane.
geometry_shader = """
#version 430

uniform mat4 view_projection_matrix;

layout(points) in;
layout(triangle_strip, max_vertices=3) out;

out layout(location = 0) vec3 vertex_position;
out layout(location = 1) vec2 barycentric_coordinates;
out layout(location = 2) float triangle_index;

in int gl_PrimitiveIDIn;
layout(binding=0) buffer triangular_mesh { float mesh_buffer[]; };


vec3 get_vertex_position(int vertex_index) {
  // Triangles are packed as 3 consecuitve vertices, each with 3 coordinates.
  int offset = gl_PrimitiveIDIn * 9 + vertex_index * 3;
  return vec3(mesh_buffer[offset], mesh_buffer[offset + 1],
    mesh_buffer[offset + 2]);
}

// Note that this function can cause artifacts for triangles that cross the eye
// plane.
bool is_back_facing(vec4 projected_vertex_0, vec4 projected_vertex_1,
                    vec4 projected_vertex_2) {
  projected_vertex_0 /= projected_vertex_0.w;
  projected_vertex_1 /= projected_vertex_1.w;
  projected_vertex_2 /= projected_vertex_2.w;
  vec2 a = (projected_vertex_1.xy - projected_vertex_0.xy);
  vec2 b = (projected_vertex_2.xy - projected_vertex_0.xy);
  return (a.x * b.y - b.x * a.y) <= 0;
}

void main() {
  vec3 positions[3] = {get_vertex_position(0), get_vertex_position(1),
                        get_vertex_position(2)};
  vec4 projected_vertices[3] = {
                            view_projection_matrix * vec4(positions[0], 1.0),
                            view_projection_matrix * vec4(positions[1], 1.0),
                            view_projection_matrix * vec4(positions[2], 1.0)};

  // Cull back-facing triangles.
  if (is_back_facing(projected_vertices[0], projected_vertices[1],
      projected_vertices[2])) {
    return;
  }

  for (int i = 0; i < 3; ++i) {
    // gl_Position is a pre-defined size 4 output variable.
    gl_Position = projected_vertices[i];
    barycentric_coordinates = vec2(i==0 ? 1.0 : 0.0, i==1 ? 1.0 : 0.0);
    triangle_index = gl_PrimitiveIDIn;

    vertex_position = positions[i];
    EmitVertex();
  }
  EndPrimitive();
}
"""

# TODO(b/151133955): add support to render a foreground / background mask.

# Fragment shader that packs barycentric coordinates, triangle index, and depth
# map in a resulting vec4 per pixel.
fragment_shader = """
#version 430

in layout(location = 0) vec3 vertex_position;
in layout(location = 1) vec2 barycentric_coordinates;
in layout(location = 2) float triangle_index;

out vec4 output_color;

void main() {
  output_color = vec4(round(triangle_index), 0.0, 0.0, 0.0);
}
"""


class TriangleRasterizer(object):
  """A class allowing to rasterize triangular meshes.

  The resulting images contain perspective-correct interpolation of attributes
  defined at the vertices of the rasterized meshes. Attributes can be defined as
  arbitrary K-dimensional values, which include depth, appearance,
  'neural features', etc.
  """

  # TODO(b/152064210): Auto-generate the default background geometry.
  def __init__(self,
               background_vertices,
               background_attributes,
               background_triangles,
               camera_origin,
               look_at,
               camera_up,
               field_of_view,
               image_size,
               near_plane,
               far_plane,
               bottom_left=(0.0, 0.0),
               name=None):
    """Initializes TriangleRasterizer with OpenGL parameters and the background.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      background_vertices: A tensor of shape `[V, 3]` containing `V` 3D
        vertices. Note that these background vertices will be used in every
        rasterized image.
      background_attributes: A tensor of shape `[V, K]` containing `V` vertices
        associated with K-dimensional attributes. Pixels for which the first
        visible surface is in the background geometry will make use of
        `background_attribute` for estimating their own attribute. Note that
        these background attributes will be use in every rasterized image.
      background_triangles: An integer tensor of shape `[T, 3]` containing `T`
        triangles, each associated with 3 vertices from `background_vertices`.
        Note that these background triangles will be used in every rasterized
        image.
      camera_origin: A Tensor of shape `[A1, ..., An, 3]`, where the last axis
        represents the 3D position of the camera.
      look_at: A Tensor of shape `[A1, ..., An, 3]`, with the last axis storing
        the position where the camera is looking at.
      camera_up: A Tensor of shape `[A1, ..., An, 3]`, where the last axis
        defines the up vector of the camera.
      field_of_view:  A Tensor of shape `[A1, ..., An, 1]`, where the last axis
        represents the vertical field of view of the frustum expressed in
        radians. Note that values for `field_of_view` must be in the range (0,
        pi).
      image_size: A tuple (height, width) containing the dimensions in pixels of
        the rasterized image".
      near_plane: A Tensor of shape `[A1, ..., An, 1]`, where the last axis
        captures the distance between the viewer and the near clipping plane.
        Note that values for `near_plane` must be non-negative.
      far_plane: A Tensor of shape `[A1, ..., An, 1]`, where the last axis
        captures the distance between the viewer and the far clipping plane.
        Note that values for `far_plane` must be non-negative.
      bottom_left: A Tensor of shape `[A1, ..., An, 2]`, where the last axis
        captures the position (in pixels) of the lower left corner of the
        screen. Defaults to (0.0, 0.0).
      name: A name for this op. Defaults to 'triangle_rasterizer_init'.
    """
    with tf.compat.v1.name_scope(
        name, "triangle_rasterizer_init",
        (background_vertices, background_attributes, background_triangles,
         camera_origin, look_at, camera_up, field_of_view, near_plane,
         far_plane, bottom_left)):

      background_vertices = tf.convert_to_tensor(value=background_vertices)
      background_attributes = tf.convert_to_tensor(value=background_attributes)
      background_triangles = tf.convert_to_tensor(value=background_triangles)

      shape.check_static(
          tensor=background_vertices,
          tensor_name="background_vertices",
          has_rank=2,
          has_dim_equals=(-1, 3))
      shape.check_static(
          tensor=background_attributes,
          tensor_name="background_attributes",
          has_rank=2)
      shape.check_static(
          tensor=background_triangles,
          tensor_name="background_triangles",
          # has_rank=2,
          has_dim_equals=(-1, 3))
      shape.compare_batch_dimensions(
          tensors=(background_vertices, background_attributes),
          last_axes=-2,
          tensor_names=("background_geometry", "background_attribute"),
          broadcast_compatible=False)

      background_vertices = tf.expand_dims(background_vertices, axis=0)
      background_attributes = tf.expand_dims(background_attributes, axis=0)

      height = float(image_size[0])
      width = float(image_size[1])

      self._background_geometry = tf.gather(
          background_vertices, background_triangles, axis=-2)
      self._background_attribute = tf.gather(
          background_attributes, background_triangles, axis=-2)

      self._camera_origin = tf.convert_to_tensor(value=camera_origin)
      self._look_at = tf.convert_to_tensor(value=look_at)
      self._camera_up = tf.convert_to_tensor(value=camera_up)
      self._field_of_view = tf.convert_to_tensor(value=field_of_view)
      self._image_size_glm = tf.convert_to_tensor(value=(width, height))
      self._image_size_int = (int(width), int(height))
      self._near_plane = tf.convert_to_tensor(value=near_plane)
      self._far_plane = tf.convert_to_tensor(value=far_plane)
      self._bottom_left = tf.convert_to_tensor(value=bottom_left)

      # Construct the pixel grid. Note that OpenGL uses half-integer pixel
      # centers.
      px = tf.linspace(0.5, width - 0.5, num=int(width))
      py = tf.linspace(0.5, height - 0.5, num=int(height))
      xv, yv = tf.meshgrid(px, py)
      self._pixel_position = tf.stack((xv, yv), axis=-1)

      # Construct the view projection matrix.
      world_to_camera = glm.look_at_right_handed(camera_origin, look_at,
                                                 camera_up)
      perspective_matrix = glm.perspective_right_handed(field_of_view,
                                                        (width / height,),
                                                        near_plane, far_plane)
      perspective_matrix = tf.squeeze(perspective_matrix)
      self._view_projection_matrix = tf.linalg.matmul(perspective_matrix,
                                                      world_to_camera)

  def rasterize(self,
                scene_vertices=None,
                scene_attributes=None,
                scene_triangles=None,
                name=None):
    """Rasterizes the scene.

    This rasterizer estimates which triangle is associated with each pixel using
    OpenGL. Then the value of attributes are estimated using Tensorflow,
    allowing to get gradients flowing through the attributes. Attributes can be
    depth, appearance, or more generally, any K-dimensional representation. Note
    that similarly to algorithms like Iterative Closest Point (ICP), not having
    gradients through correspondence does not prevent from optimizing the scene
    geometry. Custom gradients can be defined to alleviate this property.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      scene_vertices: A tensor of shape `[A1, ..., An, V, 3]` containing batches
        of `V` vertices, each defined by a 3D point.
      scene_attributes: A tensor of shape `[A1, ..., An, V, K]` containing
        batches of `V` vertices, each associated with K-dimensional attributes.
      scene_triangles: A tensor of shape `[T, 3]` containing `T` triangles, each
        associated with 3 vertices from `scene_vertices`
      name: A name for this op. Defaults to 'triangle_rasterizer_rasterize'.

    Returns:
      A tensor of shape `[A1, ..., An, H, W, K]` containing batches of images of
      height `H` and width `W`, where each pixel contains attributes rasterized
      from the scene.
    """
    with tf.compat.v1.name_scope(
        name, "triangle_rasterizer_rasterize",
        (scene_vertices, scene_attributes, scene_triangles)):
      scene_vertices = tf.convert_to_tensor(value=scene_vertices)
      scene_attributes = tf.convert_to_tensor(value=scene_attributes)
      scene_triangles = tf.convert_to_tensor(value=scene_triangles)

      shape.check_static(
          tensor=scene_vertices,
          tensor_name="scene_vertices",
          has_rank_greater_than=1,
          has_dim_equals=((-1, 3)))
      shape.compare_batch_dimensions(
          tensors=(scene_vertices, scene_attributes),
          last_axes=-2,
          tensor_names=("vertex_positions", "vertex_attributes"),
          broadcast_compatible=False)
      shape.check_static(
          tensor=scene_triangles,
          tensor_name="scene_triangles",
          has_dim_equals=((-1, 3)))

      batch_dims_triangles = len(scene_triangles.shape[:-2])
      scene_attributes = tf.gather(
          scene_attributes,
          scene_triangles,
          axis=-2,
          batch_dims=batch_dims_triangles)
      scene_geometry = tf.gather(
          scene_vertices,
          scene_triangles,
          axis=-2,
          batch_dims=batch_dims_triangles)

      batch_shape = scene_geometry.shape[:-3]
      batch_shape = [_dim_value(dim) for dim in batch_shape]

      background_geometry = tf.broadcast_to(
          self._background_geometry,
          batch_shape + self._background_geometry.shape)
      background_attribute = tf.broadcast_to(
          self._background_attribute,
          batch_shape + self._background_attribute.shape)
      geometry = tf.concat((background_geometry, scene_geometry), axis=-3)
      attributes = tf.concat((background_attribute, scene_attributes), axis=-3)

      view_projection_matrix = tf.broadcast_to(
          input=self._view_projection_matrix,
          shape=batch_shape + self._view_projection_matrix.shape)
      rasterized_face = render_ops.rasterize(
          num_points=geometry.shape[-3],
          variable_names=("view_projection_matrix", "triangular_mesh"),
          variable_kinds=("mat", "buffer"),
          variable_values=(view_projection_matrix,
                           tf.reshape(geometry, shape=batch_shape + [-1])),
          output_resolution=self._image_size_int,
          vertex_shader=vertex_shader,
          geometry_shader=geometry_shader,
          fragment_shader=fragment_shader)
      triangle_index = tf.cast(rasterized_face[..., 0], tf.int32)
      vertices_per_pixel = tf.gather(
          geometry, triangle_index, axis=-3, batch_dims=len(batch_shape))
      attributes_per_pixel = tf.gather(
          attributes, triangle_index, axis=-3, batch_dims=len(batch_shape))
      return glm.perspective_correct_interpolation(
          vertices_per_pixel, attributes_per_pixel, self._pixel_position,
          self._camera_origin, self._look_at, self._camera_up,
          self._field_of_view, self._image_size_glm, self._near_plane,
          self._far_plane, self._bottom_left)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
