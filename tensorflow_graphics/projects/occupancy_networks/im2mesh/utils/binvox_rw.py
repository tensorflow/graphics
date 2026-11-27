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
""" NO COMMENT NOW"""


#  Copyright (C) 2012 Daniel Maturana
#  This file is part of binvox-rw-py.
#
#  binvox-rw-py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  binvox-rw-py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with binvox-rw-py. If not, see <http://www.gnu.org/licenses/>.
#
#  Modified by Christopher B. Choy <chrischoy at ai dot stanford dot edu>
#  for python 3 support

import numpy as np

"""
Binvox to Numpy and back.


>>> import numpy as np
>>> import binvox_rw
>>> with open('chair.binvox', 'rb') as f:
...     m1 = binvox_rw.read_as_3d_array(f)
...
>>> m1.dims
[32, 32, 32]
>>> m1.scale
41.133000000000003
>>> m1.translate
[0.0, 0.0, 0.0]
>>> with open('chair_out.binvox', 'wb') as f:
...     m1.write(f)
...
>>> with open('chair_out.binvox', 'rb') as f:
...     m2 = binvox_rw.read_as_3d_array(f)
...
>>> m1.dims==m2.dims
True
>>> m1.scale==m2.scale
True
>>> m1.translate==m2.translate
True
>>> np.all(m1.data==m2.data)
True

>>> with open('chair.binvox', 'rb') as f:
...     md = binvox_rw.read_as_3d_array(f)
...
>>> with open('chair.binvox', 'rb') as f:
...     ms = binvox_rw.read_as_coord_array(f)
...
>>> data_ds = binvox_rw.dense_to_sparse(md.data)
>>> data_sd = binvox_rw.sparse_to_dense(ms.data, 32)
>>> np.all(data_sd==md.data)
True
>>> # the ordering of elements returned by numpy.nonzero changes with axis
>>> # ordering, so to compare for equality we first lexically sort the voxels.
>>> np.all(ms.data[:, np.lexsort(ms.data)] == data_ds[:, np.lexsort(data_ds)])
True
"""


class Voxels(object):
  """ Holds a binvox model.
  data is either a three-dimensional numpy boolean array (dense representation)
  or a two-dimensional numpy float array (coordinate representation).

  dims, translate and scale are the model metadata.

  dims are the voxel dimensions, e.g. [32, 32, 32] for a 32x32x32 model.

  scale and translate relate the voxels to the original model coordinates.

  To translate voxel coordinates i, j, k to original coordinates x, y, z:

  x_n = (i+.5)/dims[0]
  y_n = (j+.5)/dims[1]
  z_n = (k+.5)/dims[2]
  x = scale*x_n + translate[0]
  y = scale*y_n + translate[1]
  z = scale*z_n + translate[2]

  """

  def __init__(self, data, dims, translate, scale, axis_order):
    self.data = data
    self.dims = dims
    self.translate = translate
    self.scale = scale
    assert axis_order in ('xzy', 'xyz')
    self.axis_order = axis_order

  def clone(self):
    data = self.data.copy()
    dims = self.dims[:]
    translate = self.translate[:]
    return Voxels(data, dims, translate, self.scale, self.axis_order)

  def write(self, fp):
    write(self, fp)


def read_header(fp):
  """ Read binvox header. Mostly meant for internal use.
  """
  line = fp.readline().strip()
  if not line.startswith(b'#binvox'):
    raise IOError('Not a binvox file')
  dims = [int(i) for i in fp.readline().strip().split(b' ')[1:]]
  translate = [float(i) for i in fp.readline().strip().split(b' ')[1:]]
  scale = [float(i) for i in fp.readline().strip().split(b' ')[1:]][0]
  line = fp.readline()
  return dims, translate, scale


def read_as_3d_array(fp, fix_coords=True):
  """ Read binary binvox format as array.

  Returns the model with accompanying metadata.

  Voxels are stored in a three-dimensional numpy array, which is simple and
  direct, but may use a lot of memory for large models. (Storage requirements
  are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
  boolean arrays use a byte per element).

  Doesn't do any checks on input except for the '#binvox' line.
  """
  dims, translate, scale = read_header(fp)
  raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
  # if just using reshape() on the raw data:
  # indexing the array as array[i,j,k], the indices map into the
  # coords as:
  # i -> x
  # j -> z
  # k -> y
  # if fix_coords is true, then data is rearranged so that
  # mapping is
  # i -> x
  # j -> y
  # k -> z
  values, counts = raw_data[::2], raw_data[1::2]
  data = np.repeat(values, counts).astype(np.bool)
  data = data.reshape(dims)
  if fix_coords:
    # xzy to xyz TODO the right thing
    data = np.transpose(data, (0, 2, 1))
    axis_order = 'xyz'
  else:
    axis_order = 'xzy'
  return Voxels(data, dims, translate, scale, axis_order)


def read_as_coord_array(fp, fix_coords=True):
  """ Read binary binvox format as coordinates.

  Returns binvox model with voxels in a "coordinate" representation, i.e.  an
  3 x N array where N is the number of nonzero voxels. Each column
  corresponds to a nonzero voxel and the 3 rows are the (x, z, y) coordinates
  of the voxel.  (The odd ordering is due to the way binvox format lays out
  data).  Note that coordinates refer to the binvox voxels, without any
  scaling or translation.

  Use this to save memory if your model is very sparse (mostly empty).

  Doesn't do any checks on input except for the '#binvox' line.
  """
  dims, translate, scale = read_header(fp)
  raw_data = np.frombuffer(fp.read(), dtype=np.uint8)

  values, counts = raw_data[::2], raw_data[1::2]

  index, end_index = 0, 0
  end_indices = np.cumsum(counts)
  indices = np.concatenate(([0], end_indices[:-1])).astype(end_indices.dtype)

  values = values.astype(np.bool)
  indices = indices[values]
  end_indices = end_indices[values]

  nz_voxels = []
  for index, end_index in zip(indices, end_indices):
    nz_voxels.extend(range(index, end_index))
  nz_voxels = np.array(nz_voxels)
  # TODO are these dims correct?
  # according to docs,
  # index = x * wxh + z * width + y; // wxh = width * height = d * d

  x = nz_voxels / (dims[0]*dims[1])
  zwpy = nz_voxels % (dims[0]*dims[1])  # z*w + y
  z = zwpy / dims[0]
  y = zwpy % dims[0]
  if fix_coords:
    data = np.vstack((x, y, z))
    axis_order = 'xyz'
  else:
    data = np.vstack((x, z, y))
    axis_order = 'xzy'

  # return Voxels(data, dims, translate, scale, axis_order)
  return Voxels(np.ascontiguousarray(data), dims, translate, scale, axis_order)


def dense_to_sparse(voxel_data, dtype=np.int):
  """ From dense representation to sparse (coordinate) representation.
  No coordinate reordering.
  """
  if voxel_data.ndim != 3:
    raise ValueError('voxel_data is wrong shape; should be 3D array.')
  return np.asarray(np.nonzero(voxel_data), dtype)


def sparse_to_dense(voxel_data, dims, dtype=np.bool):
  """
  TODO: NO DOC NOW
  """
  if voxel_data.ndim != 2 or voxel_data.shape[0] != 3:
    raise ValueError('voxel_data is wrong shape; should be 3xN array.')
  if np.isscalar(dims):
    dims = [dims]*3
  dims = np.atleast_2d(dims).T
  # truncate to integers
  xyz = voxel_data.astype(np.int)
  # discard voxels that fall outside dims
  valid_ix = ~np.any((xyz < 0) | (xyz >= dims), 0)
  xyz = xyz[:, valid_ix]
  out = np.zeros(dims.flatten(), dtype=dtype)
  out[tuple(xyz)] = True
  return out

# def get_linear_index(x, y, z, dims):
  # """ Assuming xzy order. (y increasing fastest.
  # TODO ensure this is right when dims are not all same
  # """
  # return x*(dims[1]*dims[2]) + z*dims[1] + y


def write(voxel_model, fp):
  """ Write binary binvox format.

  Note that when saving a model in sparse (coordinate) format, it is first
  converted to dense format.

  Doesn't check if the model is 'sane'.

  """
  if voxel_model.data.ndim == 2:
    # TODO avoid conversion to dense
    dense_voxel_data = sparse_to_dense(voxel_model.data, voxel_model.dims)
  else:
    dense_voxel_data = voxel_model.data

  fp.write('#binvox 1\n')
  fp.write('dim '+' '.join(map(str, voxel_model.dims))+'\n')
  fp.write('translate '+' '.join(map(str, voxel_model.translate))+'\n')
  fp.write('scale '+str(voxel_model.scale)+'\n')
  fp.write('data\n')
  if not voxel_model.axis_order in ('xzy', 'xyz'):
    raise ValueError('Unsupported voxel model axis order')

  if voxel_model.axis_order == 'xzy':
    voxels_flat = dense_voxel_data.flatten()
  elif voxel_model.axis_order == 'xyz':
    voxels_flat = np.transpose(dense_voxel_data, (0, 2, 1)).flatten()

  # keep a sort of state machine for writing run length encoding
  state = voxels_flat[0]
  ctr = 0
  for c in voxels_flat:
    if c == state:
      ctr += 1
      # if ctr hits max, dump
      if ctr == 255:
        fp.write(chr(state))
        fp.write(chr(ctr))
        ctr = 0
    else:
      # if switch state, dump
      fp.write(chr(state))
      fp.write(chr(ctr))
      state = c
      ctr = 1
  # flush out remainders
  if ctr > 0:
    fp.write(chr(state))
    fp.write(chr(ctr))


if __name__ == '__main__':
  import doctest
  doctest.testmod()
