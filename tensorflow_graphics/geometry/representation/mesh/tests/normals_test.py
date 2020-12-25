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
"""Tests for normals."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation.mesh import normals
from tensorflow_graphics.util import test_case


class MeshTest(test_case.TestCase):

  @parameterized.parameters(
      (((None, 3), (None, 3)), (tf.float32, tf.int32)),
      (((3, 6, 3), (3, 5, 4)), (tf.float32, tf.int32)),
  )
  def test_gather_faces_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(normals.gather_faces, shapes, dtypes)

  @parameterized.parameters(
      ("Not all batch dimensions are identical", (3, 5, 4, 4), (1, 2, 4, 4)),
      ("Not all batch dimensions are identical", (5, 4, 4), (1, 2, 4, 4)),
      ("Not all batch dimensions are identical", (3, 5, 4, 4), (2, 4, 4)),
      ("vertices must have a rank greater than 1", (4,), (1, 2, 4, 4)),
      ("indices must have a rank greater than 1", (3, 5, 4, 4), (4,)),
  )
  def test_gather_faces_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(normals.gather_faces, error_msg, shapes)

  def test_gather_faces_jacobian_random(self):
    """Test the Jacobian of the face extraction function."""
    tensor_size = np.random.randint(2, 5)
    tensor_shape = np.random.randint(1, 5, size=tensor_size).tolist()
    vertex_init = np.random.random(size=tensor_shape)
    indices_init = np.random.randint(0, tensor_shape[-2], size=tensor_shape)
    indices_tensor = tf.convert_to_tensor(value=indices_init)

    def gather_faces(vertex_tensor):
      return normals.gather_faces(vertex_tensor, indices_tensor)

    self.assert_jacobian_is_correct_fn(gather_faces, [vertex_init])

  @parameterized.parameters(
      ((((0.,), (1.,)), ((1, 0),)), ((((1.,), (0.,)),),)),
      ((((0., 1.), (2., 3.)), ((1, 0),)), ((((2., 3.), (0., 1.)),),)),
      ((((0., 1., 2.), (3., 4., 5.)), ((1, 0),)), ((((3., 4., 5.),
                                                     (0., 1., 2.)),),)),
  )
  def test_gather_faces_preset(self, test_inputs, test_outputs):
    """Tests the extraction of mesh faces."""
    self.assert_output_is_correct(
        normals.gather_faces, test_inputs, test_outputs, tile=False)

  def test_gather_faces_random(self):
    """Tests the extraction of mesh faces."""
    tensor_size = np.random.randint(3, 5)
    tensor_shape = np.random.randint(1, 5, size=tensor_size).tolist()
    vertices = np.random.random(size=tensor_shape)
    indices = np.arange(tensor_shape[-2])
    indices = indices.reshape([1] * (tensor_size - 1) + [-1])
    indices = np.tile(indices, tensor_shape[:-2] + [1, 1])
    expected = np.expand_dims(vertices, -3)

    self.assertAllClose(
        normals.gather_faces(vertices, indices), expected, rtol=1e-3)

  @parameterized.parameters(
      (((None, 4, 3),), (tf.float32,)),
      (((4, 3),), (tf.float32,)),
      (((3, 4, 3),), (tf.float32,)),
  )
  def test_face_normals_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(normals.face_normals, shapes, dtypes)

  @parameterized.parameters(
      ("faces must have a rank greater than 1.", (3,)),
      ("faces must have greater than 2 dimensions in axis -2", (2, 3)),
      ("faces must have exactly 3 dimensions in axis -1.", (5, 2)),
  )
  def test_face_normals_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(normals.face_normals, error_msg, shapes)

  def test_face_normals_jacobian_random(self):
    """Test the Jacobian of the face normals function."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    tensor_vertex_shape = list(tensor_out_shape)
    tensor_vertex_shape[-1] *= 3
    tensor_index_shape = tensor_out_shape[-1]
    vertex_init = np.random.random(size=tensor_vertex_shape + [3])
    index_init = np.arange(tensor_vertex_shape[-1])
    np.random.shuffle(index_init)
    index_init = np.reshape(index_init, newshape=[1] * \
                            (tensor_vertex_size - 1) + \
                            [tensor_index_shape, 3])
    index_init = np.tile(index_init, tensor_vertex_shape[:-1] + [1, 1])
    index_tensor = tf.convert_to_tensor(value=index_init)

    def face_normals(vertex_tensor):
      face_tensor = normals.gather_faces(vertex_tensor, index_tensor)
      return normals.face_normals(face_tensor)

    self.assert_jacobian_is_correct_fn(
        face_normals, [vertex_init], atol=1e-4, delta=1e-9)

  @parameterized.parameters(
      ((((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 1, 2),)),
       (((0., 0., 1.),),)),
      ((((0., 0., 0.), (0., 0., 1.), (1., 0., 0.)), ((0, 1, 2),)),
       (((0., 1., 0.),),)),
      ((((0., 0., 0.), (0., 1., 0.), (0., 0., 1.)), ((0, 1, 2),)),
       (((1., 0., 0.),),)),
      ((((0., -2., -2.), (0, -2., 2.), (0., 2., 2.), (0., 2., -2.)),
        ((0, 1, 2, 3),)), (((-1., 0., 0.),),)),
  )
  def test_face_normals_preset(self, test_inputs, test_outputs):
    """Tests the computation of mesh face normals."""
    faces = normals.gather_faces(*test_inputs[:2])
    test_inputs = [faces] + list(test_inputs[2:])

    self.assert_output_is_correct(
        normals.face_normals, test_inputs, test_outputs, tile=False)

  def test_face_normals_random(self):
    """Tests the computation of mesh face normals in each axis."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    tensor_vertex_shape = list(tensor_out_shape)
    tensor_vertex_shape[-1] *= 3
    tensor_index_shape = tensor_out_shape[-1]

    for i in range(3):
      vertices = np.random.random(size=tensor_vertex_shape + [3])
      indices = np.arange(tensor_vertex_shape[-1])
      np.random.shuffle(indices)
      indices = np.reshape(indices,
                           newshape=[1] * (tensor_vertex_size - 1) \
                           + [tensor_index_shape, 3])
      indices = np.tile(indices, tensor_vertex_shape[:-1] + [1, 1])
      vertices[..., i] = 0.
      expected = np.zeros(shape=tensor_out_shape + [3], dtype=vertices.dtype)
      expected[..., i] = 1.
      faces = normals.gather_faces(vertices, indices)

      self.assertAllClose(
          tf.abs(normals.face_normals(faces)), expected, rtol=1e-3)

  @parameterized.parameters(
      (((4, 3), (5, 3)), (tf.float32, tf.int32)),
      (((None, 3), (None, 3)), (tf.float32, tf.int32)),
      (((3, None, 3), (3, None, 5)), (tf.float32, tf.int32)),
      (((3, 6, 3), (3, 5, 5)), (tf.float32, tf.int32)),
  )
  def test_vertex_normals_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(normals.vertex_normals, shapes, dtypes)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible.", (3, 5, 4, 3),
       (1, 2, 4, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 200, 3),
       (4, 100, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (5, 4, 3),
       (1, 2, 4, 3)),
      ("Not all batch dimensions are broadcast-compatible.", (3, 5, 4, 3),
       (2, 4, 3)),
      ("vertices must have a rank greater than 1.", (3,), (1, 2, 4, 3)),
      ("indices must have a rank greater than 1.", (3, 5, 4, 3), (3,)),
      ("vertices must have exactly 3 dimensions in axis -1.", (3, 5, 4, 2),
       (3, 5, 4, 3)),
      ("indices must have greater than 2 dimensions in axis -1.", (3, 5, 4, 3),
       (3, 5, 4, 2)),
      ("'indices' must have specified batch dimensions.", (None, 6, 3),
       (None, 5, 5)),
  )
  def test_vertex_normals_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(normals.vertex_normals, error_msg, shapes)

  def test_vertex_normals_jacobian_random(self):
    """Test the Jacobian of the vertex normals function."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()
    vertex_axis = np.array(((0., 0., 1), (1., 0., 0.), (0., 1., 0.),
                            (0., 0., -1.), (-1., 0., 0.), (0., -1., 0.)),
                           dtype=np.float32)
    vertex_axis = vertex_axis.reshape([1] * tensor_vertex_size + [6, 3])
    faces = np.array(((0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 1), (3, 2, 1),
                      (3, 4, 2), (3, 5, 4), (3, 1, 5)),
                     dtype=np.int32)
    faces = faces.reshape([1] * tensor_vertex_size + [8, 3])
    index_init = np.tile(faces, tensor_out_shape + [1, 1])
    vertex_scale = np.random.uniform(0.5, 5., tensor_out_shape + [1] * 2)
    vertex_init = vertex_axis * vertex_scale
    index_tensor = tf.convert_to_tensor(value=index_init)

    def vertex_normals(vertex_tensor):
      return normals.vertex_normals(vertex_tensor, index_tensor)

    self.assert_jacobian_is_correct_fn(vertex_normals, [vertex_init])

  @parameterized.parameters(
      (((((-1., -1., 1.), (-1., 1., 1.), (-1., -1., -1.), (-1., 1., -1.),
          (1., -1., 1.), (1., 1., 1.), (1., -1., -1.), (1., 1., -1.)),),
        (((1, 2, 0), (3, 6, 2), (7, 4, 6), (5, 0, 4), (6, 0, 2), (3, 5, 7),
          (1, 3, 2), (3, 7, 6), (7, 5, 4), (5, 1, 0), (6, 4, 0), (3, 1, 5)),)),
       ((((-0.3333333134651184, -0.6666666269302368, 0.6666666269302368),
          (-0.8164965510368347, 0.40824827551841736, 0.40824827551841736),
          (-0.8164965510368347, -0.40824827551841736, -0.40824827551841736),
          (-0.3333333134651184, 0.6666666269302368, -0.6666666269302368),
          (0.8164965510368347, -0.40824827551841736, 0.40824827551841736),
          (0.3333333134651184, 0.6666666269302368, 0.6666666269302368),
          (0.3333333134651184, -0.6666666269302368, -0.6666666269302368),
          (0.8164965510368347, 0.40824827551841736, -0.40824827551841736)),),)),
  )
  def test_vertex_normals_preset(self, test_inputs, test_outputs):
    """Tests the computation of vertex normals."""
    self.assert_output_is_correct(
        normals.vertex_normals, test_inputs, test_outputs, tile=False)

  def test_vertex_normals_random(self):
    """Tests the computation of vertex normals for a regular octahedral."""
    tensor_vertex_size = np.random.randint(1, 3)
    tensor_out_shape = np.random.randint(1, 5, size=tensor_vertex_size)
    tensor_out_shape = tensor_out_shape.tolist()

    with self.subTest(name="triangular_faces"):
      vertex_on_axes = np.array(((0., 0., 1), (1., 0., 0.), (0., 1., 0.),
                                 (0., 0., -1.), (-1., 0., 0.), (0., -1., 0.)),
                                dtype=np.float32)
      vertex_on_axes = vertex_on_axes.reshape([1] * tensor_vertex_size + [6, 3])
      index_init = np.array(((0, 1, 2), (0, 2, 4), (0, 4, 5), (0, 5, 1),
                             (3, 2, 1), (3, 4, 2), (3, 5, 4), (3, 1, 5)),
                            dtype=np.int32)
      index_init = index_init.reshape([1] * tensor_vertex_size + [8, 3])
      index_init = np.tile(index_init, tensor_out_shape + [1, 1])
      vertex_scale = np.random.uniform(0.5, 5., tensor_out_shape + [1] * 2)
      vertex_init = vertex_on_axes * vertex_scale
      expected = vertex_on_axes * (vertex_scale * 0. + 1.)

      vertex_tensor = tf.convert_to_tensor(value=vertex_init)
      index_tensor = tf.convert_to_tensor(value=index_init)

      self.assertAllClose(
          normals.vertex_normals(vertex_tensor, index_tensor), expected)

    with self.subTest(name="polygon_faces"):
      num_vertices = np.random.randint(4, 8)
      poly_vertices = []
      rad_step = np.pi * 2. / num_vertices
      for i in range(num_vertices):
        poly_vertices.append([np.cos(i * rad_step), np.sin(i * rad_step), 0])
      vertex_init = np.array(poly_vertices, dtype=np.float32)
      vertex_init = vertex_init.reshape([1] * tensor_vertex_size + [-1, 3])
      vertex_init = vertex_init * vertex_scale
      index_init = np.arange(num_vertices, dtype=np.int32)
      index_init = index_init.reshape([1] * tensor_vertex_size + [1, -1])
      index_init = np.tile(index_init, tensor_out_shape + [1, 1])
      expected = np.array((0., 0., 1.), dtype=np.float32)
      expected = expected.reshape([1] * tensor_vertex_size + [1, 3])
      expected = np.tile(expected, tensor_out_shape + [num_vertices, 1])
      vertex_tensor = tf.convert_to_tensor(value=vertex_init)
      index_tensor = tf.convert_to_tensor(value=index_init)

      self.assertAllClose(
          normals.vertex_normals(vertex_tensor, index_tensor), expected)


if __name__ == "__main__":
  test_case.main()
