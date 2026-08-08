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
"""Class to represent a point cloud."""

import tensorflow as tf
from tensorflow_graphics.geometry.convolution.utils import \
    flatten_batch_to_2d, unflatten_2d_to_batch

from pylib.pc.utils import check_valid_point_cloud_input


class _AABB:
  """Axis aligned bounding box of a point cloud.

  Args:
    Pointcloud: A 'PointCloud' instance from which to compute the
      axis aligned bounding box.

  """

  def __init__(self, point_cloud, name=None):

    self._batch_size = point_cloud._batch_size
    self._batch_shape = point_cloud._batch_shape
    self.point_cloud_ = point_cloud

    self._aabb_min = tf.math.unsorted_segment_min(
        data=point_cloud._points, segment_ids=point_cloud._batch_ids,
        num_segments=self._batch_size) - 1e-9
    self._aabb_max = tf.math.unsorted_segment_max(
        data=point_cloud._points, segment_ids=point_cloud._batch_ids,
        num_segments=self._batch_size) + 1e-9

  def get_diameter(self, ord='euclidean', name=None):
    """ Returns the diameter of the bounding box.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      ord: Order of the norm. Supported values are `'euclidean'`,
          `1`, `2`, `np.inf` and any positive real number yielding the
          corresponding p-norm. Default is `'euclidean'`. (optional)
    Return:
      diam: A `float` 'Tensor' of shape `[A1, ..., An]`, diameters of the
        bounding boxes

    """

    diam = tf.linalg.norm(self._aabb_max - self._aabb_min, ord=ord, axis=-1)
    if self._batch_shape is None:
      return diam
    else:
      return tf.reshape(diam, self._batch_shape)


class PointCloud:
  """ Class to represent point clouds.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      points: A float `Tensor` either of shape `[N, D]` or of shape
        `[A1, .., An, V, D]`, possibly padded as indicated by `sizes`.
        Represents the point coordinates.
      batch_ids: An `int` `Tensor` of shape `[N]` associated with the points.
        Is required if `points` is of shape `[N, D]`.
      sizes: An `int` `Tensor` of shape `[A1, ..., An]` indicating the
        true input sizes in case of padding (`sizes=None` indicates no padding)
        Note that `sizes[A1, ..., An] <= V` or `sum(sizes) == N`.
      batch_size:  An `int`, the size of the batch.
    """

  def __init__(self,
               points,
               batch_ids=None,
               batch_size=None,
               sizes=None,
               name=None):
    points = tf.convert_to_tensor(value=points, dtype=tf.float32)
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes, dtype=tf.int32)
    if batch_ids is not None:
      batch_ids = tf.convert_to_tensor(value=batch_ids, dtype=tf.int32)
    if batch_size is not None:
      self._batch_size = tf.convert_to_tensor(value=batch_size, dtype=tf.int32)
    else:
      self._batch_size = None

    check_valid_point_cloud_input(points, sizes, batch_ids)

    self._sizes = sizes
    # compatibility batch size as CPU int for graph mode
    self._batch_size_numpy = batch_size
    self._batch_ids = batch_ids
    self._dimension = tf.gather(tf.shape(points), tf.rank(points) - 1)
    self._batch_shape = None
    self._unflatten = None
    self._aabb = None

    if points.shape.ndims > 2:
      self._init_from_padded(points)
    else:
      self._init_from_segmented(points)

    if self._batch_size_numpy is None:
      self._batch_size_numpy = self._batch_size

    #Sort the points based on the batch ids in incremental order.
    self._sorted_indices_batch = tf.argsort(self._batch_ids)

  def _init_from_padded(self, points):
    """converting padded `Tensor` of shape `[A1, ..., An, V, D]` into a 2D
      `Tensor` of shape `[N,D]` with segmentation ids.
    """
    self._batch_shape = tf.shape(points)[:-2]
    if self._batch_size is None:
      self._batch_size = tf.reduce_prod(self._batch_shape)
    if self._sizes is None:
      self._sizes = tf.constant(
          value=tf.shape(points)[-2], shape=self._batch_shape)
    self._get_segment_id = tf.reshape(
        tf.range(0, self._batch_size), self._batch_shape)
    self._points, self._unflatten = flatten_batch_to_2d(points, self._sizes)
    self._batch_ids = tf.repeat(
        tf.range(0, self._batch_size),
        repeats=tf.reshape(self._sizes, [-1]))

  def _init_from_segmented(self, points):
    """if input is already 2D `Tensor` with segmentation ids or given sizes.
    """
    if self._batch_ids is None:
      if self._batch_size is None:
        self._batch_size = tf.reduce_prod(self._sizes.shape)
      self._batch_ids = tf.repeat(tf.range(0, self._batch_size), self._sizes)
    if self._batch_size is None:
      self._batch_size = tf.reduce_max(self._batch_ids) + 1
    self._points = points

  def get_points(self, id=None, max_num_points=None, name=None):
    """ Returns the points.

    Note:
      In the following, A1 to An are optional batch dimensions.

      If called withoud specifying 'id' returns the points in padded format
      `[A1, ..., An, V, D]`

    Args:
      id: An `int`, index of point cloud in the batch, if `None` returns all
      max_num_points: An `int`, specifies the 'V' dimension the method returns,
          by default uses maximum of 'sizes'. `max_rows >= max(sizes)`

    Return:
      A `float` `Tensor`
        of shape `[Ni, D]`, if 'id' was given
      or
        of shape `[A1, ..., An, V, D]`, zero padded, if no `id` was given.

    """
    if id is not None:
      if not isinstance(id, int):
        slice = self._get_segment_id
        for slice_id in id:
          slice = slice[slice_id]
        id = slice
      if id > self._batch_size:
        raise IndexError('batch index out of range')
      return self._points[self._batch_ids == id]
    else:
      return self.get_unflatten(max_num_points=max_num_points)(self._points)

  def get_sizes(self, name=None):
    """ Returns the sizes of the point clouds in the batch.

    Note:
      In the following, A1 to An are optional batch dimensions.
      Use this instead of accessing 'self._sizes',
      if the class was constructed using segmented input the '_sizes' is
      created in this method.

    Returns:
      `Tensor` of shape `[A1, .., An]`.

    """
    if self._sizes is None:
      _ids, _, self._sizes = tf.unique_with_counts(
          self._batch_ids)
      _ids_sorted = tf.argsort(_ids)
      self._sizes = tf.gather(self._sizes, _ids_sorted)
      if self._batch_shape is not None:
        self._sizes = tf.reshape(self._sizes, self._batch_shape)
    return self._sizes

  def get_unflatten(self, max_num_points=None, name=None):
    """ Returns the method to unflatten the segmented points.

    Use this instead of accessing 'self._unflatten',
    if the class was constructed using segmented input the '_unflatten' method
    is created in this method.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      max_num_points: An `int`, specifies the 'V' dimension the method returns,
        by default uses maximum of 'sizes'. `max_rows >= max(sizes)`
    Returns:
      A method to unflatten the segmented points, which returns a `Tensor` of
      shape `[A1,...,An,V,D]`, zero padded.

    Raises:
      ValueError: When trying to unflatten unsorted points.

    """
    if self._unflatten is None:
      self._unflatten = lambda data: unflatten_2d_to_batch(
          data=tf.gather(data, self._sorted_indices_batch),
          sizes=self.get_sizes(),
          max_rows=max_num_points)
    return self._unflatten

  def get_AABB(self) -> _AABB:
    """ Returns the axis aligned bounding box of the point cloud.

    Use this instead of accessing `self._aabb`, as the bounding box
    is initialized  with tthe first call of his method.

    Returns:
      A `AABB` instance

    """
    if self._aabb is None:
      self._aabb = _AABB(point_cloud=self)
    return self._aabb

  def set_batch_shape(self, batch_shape, name=None):
    """ Function to change the batch shape

      Use this to set a batch shape instead of using 'self._batch_shape' to
      also change dependent variables.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      batch_shape: A 1D `int` `Tensor` `[A1,...,An]`.

    Raises:
      ValueError: if shape does not sum up to batch size.

    """
    if batch_shape is not None:
      batch_shape = tf.convert_to_tensor(value=batch_shape, dtype=tf.int32)
      tf.assert_equal(
          tf.reduce_prod(batch_shape), self._batch_size,
          f'Incompatible batch size. Must be {self._batch_size} \
              but is {tf.reduce_prod(batch_shape)}')
      # if tf.reduce_prod(batch_shape) != self._batch_size:
      #   raise ValueError(
      #       f'Incompatible batch size. Must be {self._batch_size} \
      #        but is {tf.reduce_prod(batch_shape)}')
      self._batch_shape = batch_shape
      self._get_segment_id = tf.reshape(
          tf.range(0, self._batch_size), self._batch_shape)
      if self._sizes is not None:
        self._sizes = tf.reshape(self._sizes, self._batch_shape)
    else:
      self._batch_shape = None
