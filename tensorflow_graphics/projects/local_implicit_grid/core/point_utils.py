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
# Lint as: python3
"""Additional data utilities for point preprocessing.
"""

import numpy as np
from plyfile import PlyData
from plyfile import PlyElement


def read_point_ply(filename):
  """Load point cloud from ply file.

  Args:
    filename: str, filename for ply file to load.
  Returns:
    v: np.array of shape [#v, 3], vertex coordinates
    n: np.array of shape [#v, 3], vertex normals
  """
  pd = PlyData.read(filename)['vertex']
  v = np.array(np.stack([pd[i] for i in ['x', 'y', 'z']], axis=-1))
  n = np.array(np.stack([pd[i] for i in ['nx', 'ny', 'nz']], axis=-1))
  return v, n


def write_point_ply(filename, v, n):
  """Write point cloud to ply file.

  Args:
    filename: str, filename for ply file to load.
    v: np.array of shape [#v, 3], vertex coordinates
    n: np.array of shape [#v, 3], vertex normals
  """
  vn = np.concatenate([v, n], axis=1)
  vn = [tuple(vn[i]) for i in range(vn.shape[0])]
  vn = np.array(vn, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                           ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
  el = PlyElement.describe(vn, 'vertex')
  PlyData([el]).write(filename)


def np_pad_points(points, ntarget):
  """Pad point cloud to required size.

  If number of points is larger than ntarget, take ntarget random samples.
  If number of points is smaller than ntarget, pad by repeating last point.
  Args:
    points: `[npoints, nchannel]` np array, where first 3 channels are xyz.
    ntarget: int, number of target channels.
  Returns:
    result: `[ntarget, nchannel]` np array, padded points to ntarget numbers.
  """
  if points.shape[0] < ntarget:
    mult = np.ceil(float(ntarget)/float(points.shape[0])) - 1
    rand_pool = np.tile(points, [int(mult), 1])
    nextra = ntarget-points.shape[0]
    extra_idx = np.random.choice(rand_pool.shape[0], nextra, replace=False)
    extra_pts = rand_pool[extra_idx]
    points_out = np.concatenate([points, extra_pts], axis=0)
  else:
    idx_choice = np.random.choice(points.shape[0],
                                  size=ntarget,
                                  replace=False)
    points_out = points[idx_choice]

  return points_out


def np_gather_ijk_index(arr, index):
  arr_flat = arr.reshape(-1, arr.shape[-1])
  _, j, k, _ = arr.shape
  index_transform = index[:, 0]*j*k+index[:, 1]*k+index[:, 2]
  return arr_flat[index_transform]


def np_shifted_crop(v, idx_grid, shift, crop_size, ntarget):
  """Create a shifted crop."""
  nchannel = v.shape[1]
  vxyz = v[:, :3] - shift * crop_size * 0.5
  vall = v.copy()
  point_idxs = np.arange(v.shape[0])
  point_grid_idx = np.floor(vxyz / crop_size).astype(np.int32)
  valid_mask = np.ones(point_grid_idx.shape[0]).astype(np.bool)
  for i in range(3):
    valid_mask = np.logical_and(valid_mask, point_grid_idx[:, i] >= 0)
    valid_mask = np.logical_and(valid_mask,
                                point_grid_idx[:, i] < idx_grid.shape[i])
  point_grid_idx = point_grid_idx[valid_mask]
  # translate to global grid index
  point_grid_idx = np_gather_ijk_index(idx_grid, point_grid_idx)

  vall = vall[valid_mask]
  point_idxs = point_idxs[valid_mask]
  crop_indices, revidx = np.unique(point_grid_idx, axis=0,
                                   return_inverse=True)
  ncrops = crop_indices.shape[0]
  sortarr = np.argsort(revidx)
  revidx_sorted = revidx[sortarr]
  vall_sorted = vall[sortarr]
  point_idxs_sorted = point_idxs[sortarr]
  bins = np.searchsorted(revidx_sorted, np.arange(ncrops))
  bins = list(bins) + [v.shape[0]]
  sid = bins[0:-1]
  eid = bins[1:]
  # initialize outputs
  point_crops = np.zeros([ncrops, ntarget, nchannel])
  crop_point_idxs = []
  # extract crops and pad
  for i, (s, e) in enumerate(zip(sid, eid)):
    cropped_points = vall_sorted[s:e]
    crop_point_idx = point_idxs_sorted[s:e]
    crop_point_idxs.append(crop_point_idx)
    if cropped_points.shape[0] < ntarget:
      padded_points = np_pad_points(cropped_points, ntarget=ntarget)
    else:
      choice_idx = np.random.choice(cropped_points.shape[0],
                                    ntarget, replace=False)
      padded_points = cropped_points[choice_idx]
    point_crops[i] = padded_points
  return point_crops, crop_indices, crop_point_idxs


def np_get_occupied_idx(v,
                        xmin=(0., 0., 0.),
                        xmax=(1., 1., 1.),
                        crop_size=.125, ntarget=2048,
                        overlap=True, normalize_crops=False,
                        return_shape=False, return_crop_point_idxs=False):
  """Get crop indices for point clouds."""
  v = v.copy()-xmin
  xmin = np.array(xmin)
  xmax = np.array(xmax)
  r = (xmax-xmin)/crop_size
  r = np.ceil(r)
  rr = r.astype(np.int32) if not overlap else (2*r-1).astype(np.int32)
  # create index grid
  idx_grid = np.stack(np.meshgrid(np.arange(rr[0]),
                                  np.arange(rr[1]),
                                  np.arange(rr[2]), indexing='ij'), axis=-1)
  # [rr[0], rr[1], rr[2], 3]

  shift_idxs = np.stack(
      np.meshgrid(np.arange(int(overlap)+1),
                  np.arange(int(overlap)+1),
                  np.arange(int(overlap)+1), indexing='ij'), axis=-1)
  shift_idxs = np.reshape(shift_idxs, [-1, 3])
  point_crops = []
  crop_indices = []
  crop_point_idxs = []
  for i in range(shift_idxs.shape[0]):
    sft = shift_idxs[i]
    skp = int(overlap)+1
    idg = idx_grid[sft[0]::skp, sft[1]::skp, sft[2]::skp]
    pc, ci, cpidx = np_shifted_crop(v, idg, sft, crop_size=crop_size,
                                    ntarget=ntarget)
    point_crops.append(pc)
    crop_indices.append(ci)
    crop_point_idxs += cpidx
  point_crops = np.concatenate(point_crops, axis=0)  # [ncrops, nsurface, 6]
  crop_indices = np.concatenate(crop_indices, axis=0)  # [ncrops, 3]

  if normalize_crops:
    # normalize each crop
    crop_corners = crop_indices * 0.5 * crop_size
    crop_centers = crop_corners + 0.5 * crop_size  # [ncrops, 3]
    crop_centers = crop_centers[:, np.newaxis, :]  # [ncrops, 1, 3]
    point_crops[..., :3] = point_crops[..., :3] -crop_centers
    point_crops[..., :3] = point_crops[..., :3] / crop_size * 2

  outputs = [point_crops, crop_indices]
  if return_shape: outputs += [idx_grid.shape[:3]]
  if return_crop_point_idxs:
    outputs += [crop_point_idxs]
  return tuple(outputs)
