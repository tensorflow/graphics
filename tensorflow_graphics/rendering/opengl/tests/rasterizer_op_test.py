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
"""Tests for the opengl rasterizer op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import six
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import look_at
from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.rendering.opengl import rasterization_backend
from tensorflow_graphics.util import test_case

# Empty vertex shader
test_vertex_shader = """
#version 450
void main() { }
"""

# Geometry shader that projects the vertices of visible triangles onto the image
# plane.
test_geometry_shader = """
#version 450

uniform mat4 view_projection_matrix;

layout(points) in;
layout(triangle_strip, max_vertices=3) out;

out layout(location = 0) vec3 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 bar_coord;
out layout(location = 3) float tri_id;

layout(binding=0) buffer triangular_mesh { float mesh_buffer[]; };

vec3 get_vertex_position(int i) {
  int o = gl_PrimitiveIDIn * 9 + i * 3;
  return vec3(mesh_buffer[o + 0], mesh_buffer[o + 1], mesh_buffer[o + 2]);
}

bool is_back_facing(vec3 v0, vec3 v1, vec3 v2) {
  vec4 tv0 = view_projection_matrix * vec4(v0, 1.0);
  vec4 tv1 = view_projection_matrix * vec4(v1, 1.0);
  vec4 tv2 = view_projection_matrix * vec4(v2, 1.0);
  tv0 /= tv0.w;
  tv1 /= tv1.w;
  tv2 /= tv2.w;
  vec2 a = (tv1.xy - tv0.xy);
  vec2 b = (tv2.xy - tv0.xy);
  return (a.x * b.y - b.x * a.y) <= 0;
}

void main() {
  vec3 v0 = get_vertex_position(0);
  vec3 v1 = get_vertex_position(1);
  vec3 v2 = get_vertex_position(2);

  // Cull back-facing triangles.
  if (is_back_facing(v0, v1, v2)) {
    return;
  }

  normal = normalize(cross(v1 - v0, v2 - v0));

  vec3 positions[3] = {v0, v1, v2};
  for (int i = 0; i < 3; ++i) {
    // gl_Position is a pre-defined size 4 output variable
    gl_Position = view_projection_matrix * vec4(positions[i], 1);
    bar_coord = vec2(i==0 ? 1 : 0, i==1 ? 1 : 0);
    tri_id = gl_PrimitiveIDIn;

    position = positions[i];
    EmitVertex();
  }
  EndPrimitive();
}
"""

# Fragment shader that packs barycentric coordinates, triangle index, and depth
# map in a resulting vec4 per pixel.
test_fragment_shader = """
#version 450

in layout(location = 0) vec3 position;
in layout(location = 1) vec3 normal;
in layout(location = 2) vec2 bar_coord;
in layout(location = 3) float tri_id;

out vec4 output_color;

void main() {
  output_color = vec4(bar_coord, tri_id, position.z);
}
"""


class RasterizerOPTest(test_case.TestCase):

  def test_rasterize(self):
    max_depth = 10
    min_depth = 2
    height = 480
    width = 640
    camera_origin = (0.0, 0.0, 0.0)
    camera_up = (0.0, 1.0, 0.0)
    look_at_point = (0.0, 0.0, 1.0)
    fov = (60.0 * np.math.pi / 180,)
    near_plane = (1.0,)
    far_plane = (10.0,)
    batch_shape = tf.convert_to_tensor(
        value=(2, (max_depth - min_depth) // 2), dtype=tf.int32)

    world_to_camera = look_at.right_handed(camera_origin, look_at_point,
                                           camera_up)
    perspective_matrix = perspective.right_handed(
        fov, (float(width) / float(height),), near_plane, far_plane)
    view_projection_matrix = tf.matmul(perspective_matrix, world_to_camera)
    view_projection_matrix = tf.squeeze(view_projection_matrix)

    # Generate triangles at different depths and associated ground truth.
    tris = np.zeros((max_depth - min_depth, 9), dtype=np.float32)
    gt = np.zeros((max_depth - min_depth, height, width, 2), dtype=np.float32)
    for idx in range(max_depth - min_depth):
      tris[idx, :] = (-100.0, 100.0, idx + min_depth, 100.0, 100.0,
                      idx + min_depth, 0.0, -100.0, idx + min_depth)
      gt[idx, :, :, :] = (0, idx + min_depth)

    # Broadcast the variables.
    render_parameters = {
        "view_projection_matrix":
            ("mat",
             tf.broadcast_to(
                 input=view_projection_matrix,
                 shape=tf.concat(
                     values=(batch_shape,
                             tf.shape(input=view_projection_matrix)[-2:]),
                     axis=0))),
        "triangular_mesh":
            ("buffer",
             tf.reshape(
                 tris, shape=tf.concat(values=(batch_shape, (9,)), axis=0)))
    }
    # Reshape the ground truth.
    gt = tf.reshape(
        gt, shape=tf.concat(values=(batch_shape, (height, width, 2)), axis=0))

    render_parameters = list(six.iteritems(render_parameters))
    variable_names = [v[0] for v in render_parameters]
    variable_kinds = [v[1][0] for v in render_parameters]
    variable_values = [v[1][1] for v in render_parameters]

    def rasterize():
      return rasterization_backend.render_ops.rasterize(
          num_points=3,
          variable_names=variable_names,
          variable_kinds=variable_kinds,
          variable_values=variable_values,
          output_resolution=(width, height),
          vertex_shader=test_vertex_shader,
          geometry_shader=test_geometry_shader,
          fragment_shader=test_fragment_shader,
      )

    result = rasterize()
    self.assertAllClose(result[..., 2:4], gt)

    @tf.function
    def check_lazy_shape():
      # Within @tf.function, the tensor shape is determined by SetShapeFn
      # callback. Ensure that the shape of non-batch axes matches that of of
      # the actual tensor evaluated in eager mode above.
      lazy_shape = rasterize().shape
      self.assertEqual(lazy_shape[-3:], list(result.shape)[-3:])

    check_lazy_shape()

  @parameterized.parameters(
      ("The variable names, kinds, and values must have the same size.",
       ["var1"], ["buffer", "buffer"
                 ], [[1.0], [1.0]], tf.errors.InvalidArgumentError, ValueError),
      ("The variable names, kinds, and values must have the same size.",
       ["var1", "var2"], ["buffer"], [[1.0], [1.0]],
       tf.errors.InvalidArgumentError, ValueError),
      ("The variable names, kinds, and values must have the same size.",
       ["var1", "var2"], ["buffer", "buffer"], [[1.0]],
       tf.errors.InvalidArgumentError, ValueError),
      ("has an invalid batch", ["var1", "var2"], ["buffer", "buffer"],
       [[1.0], [[1.0]]], tf.errors.InvalidArgumentError, ValueError),
      ("has an invalid", ["var1"], ["mat"], [[1.0]],
       tf.errors.InvalidArgumentError, ValueError),
      ("has an invalid", ["var1"], ["buffer"], [1.0],
       tf.errors.InvalidArgumentError, ValueError),
  )
  def test_invalid_variable_inputs(self, error_msg, variable_names,
                                   variable_kinds, variable_values, error_eager,
                                   error_graph_mode):
    height = 1
    width = 1
    empty_shader_code = "#version 450\n void main() { }\n"
    if tf.executing_eagerly():
      error = error_eager
    else:
      error = error_graph_mode
    with self.assertRaisesRegexp(error, error_msg):
      self.evaluate(
          rasterization_backend.render_ops.rasterize(
              num_points=0,
              variable_names=variable_names,
              variable_kinds=variable_kinds,
              variable_values=variable_values,
              output_resolution=(width, height),
              vertex_shader=empty_shader_code,
              geometry_shader=empty_shader_code,
              fragment_shader=empty_shader_code))


if __name__ == "__main__":
  test_case.main()
