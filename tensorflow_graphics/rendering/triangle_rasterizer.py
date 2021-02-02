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

import tensorflow as tf

from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.rendering.opengl import math as glm
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _perspective_correct_barycentrics(vertices_per_pixel, model_to_eye_matrix,
                                      perspective_matrix, image_size_float):
  """Creates the pixels grid and computes barycentrics."""
  # Construct the pixel grid with half-integer pixel centers.
  width = image_size_float[1]
  height = image_size_float[0]
  px = tf.linspace(0.5, width - 0.5, num=int(width))
  py = tf.linspace(0.5, height - 0.5, num=int(height))
  xv, yv = tf.meshgrid(px, py)
  pixel_position = tf.stack((xv, yv), axis=-1)

  return glm.perspective_correct_barycentrics(vertices_per_pixel,
                                              pixel_position,
                                              model_to_eye_matrix,
                                              perspective_matrix,
                                              (width, height))


def _perspective_correct_attributes(attribute, barycentrics, triangles,
                                    triangle_index, len_batch_shape):
  attribute = tf.gather(attribute, triangles, axis=-2)
  attribute_per_pixel = tf.gather(
      attribute, triangle_index, axis=-3, batch_dims=len_batch_shape)

  return glm.interpolate_attributes(attribute_per_pixel, barycentrics)


def _dim_value(dim):
  return 1 if dim is None else tf.compat.v1.dimension_value(dim)


def rasterize(vertices,
              triangles,
              attributes,
              model_to_eye_matrix,
              perspective_matrix,
              image_size,
              backend=rasterization_backend.RasterizationBackends.OPENGL,
              name=None):
  """Rasterizes the scene.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    vertices: A tensor of shape `[A1, ..., An, V, 3]` containing batches of `V`
      vertices, each defined by a 3D point.
    triangles: A tensor of shape `[T, 3]` containing `T` triangles, each
      associated with 3 vertices from `vertices`.
    attributes: A dictionary of tensors, each of shape `[A1, ..., An, V, K_a]`
      containing batches of `V` vertices, each associated with K-dimensional
      attributes. K_a may vary by attribute.
    model_to_eye_matrix: A tensor of shape `[A1, ..., An, 4, 4]` containing
      batches of matrices used to transform vertices from model to eye
      coordinates.
    perspective_matrix: A tensor of shape `[A1, ..., An, 4, 4]` containing
      batches of matrices used to project vertices from eye to clip coordinates.
    image_size: A tuple (height, width) containing the dimensions in pixels of
      the rasterized image.
    backend: A rasterization_backend.RasterizationBackends enum containing the
      backend method to use for rasterization.
    name: A name for this op. Defaults to 'triangle_rasterizer_rasterize'.

  Returns:
    A dictionary. The key "mask" is of shape `[A1, ..., An, height, width, 1]`
    and stores a value of `0` of the pixel is assciated with the background,
    and `1` with the foreground. The key "barycentrics" is of shape
    `[A1, ..., An, height, width, 3]` and stores barycentric weights. Finally,
    the dictionary contains perspective correct interpolated attributes of shape
    `[A1, ..., An, height, width, K]` per entry in the `attributes` dictionary.
  """
  with tf.compat.v1.name_scope(name, "triangle_rasterizer_rasterize",
                               (vertices, triangles, attributes,
                                model_to_eye_matrix, perspective_matrix)):
    vertices = tf.convert_to_tensor(value=vertices)
    triangles = tf.convert_to_tensor(value=triangles)
    model_to_eye_matrix = tf.convert_to_tensor(value=model_to_eye_matrix)
    perspective_matrix = tf.convert_to_tensor(value=perspective_matrix)

    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_rank=2,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=model_to_eye_matrix,
        tensor_name="model_to_eye_matrix",
        has_dim_equals=(((-2, 4), (-1, 4))))
    shape.check_static(
        tensor=perspective_matrix,
        tensor_name="perspective_matrix",
        has_dim_equals=(((-2, 4), (-1, 4))))

    image_size_float = (float(image_size[0]), float(image_size[1]))
    image_size_backend = (int(image_size[1]), int(image_size[0]))

    view_projection_matrix = tf.linalg.matmul(perspective_matrix,
                                              model_to_eye_matrix)
    rasterized = rasterization_backend.rasterize(vertices, triangles,
                                                 view_projection_matrix,
                                                 image_size_backend, backend)
    outputs = {
        "mask": rasterized.foreground_mask,
        "triangle_indices": rasterized.triangle_id
    }

    # Extract batch shape in order to make sure it is preserved after `gather`
    # operation.
    batch_shape = rasterized.triangle_id.shape[:-3]
    batch_shape = [_dim_value(dim) for dim in batch_shape]

    vertices_per_pixel = tf.gather(
        vertices, rasterized.vertex_ids, batch_dims=len(batch_shape))
    barycentrics = _perspective_correct_barycentrics(vertices_per_pixel,
                                                     model_to_eye_matrix,
                                                     perspective_matrix,
                                                     image_size_float)
    mask_float = tf.cast(rasterized.foreground_mask, vertices.dtype)
    outputs["barycentrics"] = mask_float * barycentrics

    for key, attribute in attributes.items():
      attribute = tf.convert_to_tensor(value=attribute)
      outputs[key] = mask_float * _perspective_correct_attributes(
          attribute, barycentrics, triangles, rasterized.triangle_id[..., 0],
          len(batch_shape))

    return outputs


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
