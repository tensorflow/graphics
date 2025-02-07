#pykdtree, Fast kd-tree implementation with OpenMP-enabled queries
#
#Copyright (C) 2013 - present  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, int8_t, uint8_t
cimport cython


# Node structure
cdef struct node_float:
    float cut_val
    int8_t cut_dim
    uint32_t start_idx
    uint32_t n
    float cut_bounds_lv
    float cut_bounds_hv
    node_float *left_child
    node_float *right_child

cdef struct tree_float:
    float *bbox
    int8_t no_dims
    uint32_t *pidx
    node_float *root

cdef struct node_double:
    double cut_val
    int8_t cut_dim
    uint32_t start_idx
    uint32_t n
    double cut_bounds_lv
    double cut_bounds_hv
    node_double *left_child
    node_double *right_child

cdef struct tree_double:
    double *bbox
    int8_t no_dims
    uint32_t *pidx
    node_double *root

cdef extern tree_float* construct_tree_float(float *pa, int8_t no_dims, uint32_t n, uint32_t bsp) nogil
cdef extern void search_tree_float(tree_float *kdtree, float *pa, float *point_coords, uint32_t num_points, uint32_t k, float distance_upper_bound, float eps_fac, uint8_t *mask, uint32_t *closest_idxs, float *closest_dists) nogil
cdef extern void delete_tree_float(tree_float *kdtree)

cdef extern tree_double* construct_tree_double(double *pa, int8_t no_dims, uint32_t n, uint32_t bsp) nogil
cdef extern void search_tree_double(tree_double *kdtree, double *pa, double *point_coords, uint32_t num_points, uint32_t k, double distance_upper_bound, double eps_fac, uint8_t *mask, uint32_t *closest_idxs, double *closest_dists) nogil
cdef extern void delete_tree_double(tree_double *kdtree)

cdef class KDTree:
    """kd-tree for fast nearest-neighbour lookup.
    The interface is made to resemble the scipy.spatial kd-tree except
    only Euclidean distance measure is supported.

    :Parameters:
    data_pts : numpy array
        Data points with shape (n , dims)
    leafsize : int, optional
        Maximum number of data points in tree leaf
    """

    cdef tree_float *_kdtree_float
    cdef tree_double *_kdtree_double
    cdef readonly np.ndarray data_pts
    cdef readonly np.ndarray data
    cdef float *_data_pts_data_float
    cdef double *_data_pts_data_double
    cdef readonly uint32_t n
    cdef readonly int8_t ndim
    cdef readonly uint32_t leafsize

    def __cinit__(KDTree self):
        self._kdtree_float = NULL
        self._kdtree_double = NULL

    def __init__(KDTree self, np.ndarray data_pts not None, int leafsize=16):

        # Check arguments
        if leafsize < 1:
            raise ValueError('leafsize must be greater than zero')

        # Get data content
        cdef np.ndarray[float, ndim=1] data_array_float
        cdef np.ndarray[double, ndim=1] data_array_double

        if data_pts.dtype == np.float32:
            data_array_float = np.ascontiguousarray(data_pts.ravel(), dtype=np.float32)
            self._data_pts_data_float = <float *>data_array_float.data
            self.data_pts = data_array_float
        else:
            data_array_double = np.ascontiguousarray(data_pts.ravel(), dtype=np.float64)
            self._data_pts_data_double = <double *>data_array_double.data
            self.data_pts = data_array_double

        # scipy interface compatibility
        self.data = self.data_pts

        # Get tree info
        self.n = <uint32_t>data_pts.shape[0]
        self.leafsize = <uint32_t>leafsize
        if data_pts.ndim == 1:
            self.ndim = 1
        else:
            self.ndim = <int8_t>data_pts.shape[1]

        # Release GIL and construct tree
        if data_pts.dtype == np.float32:
            with nogil:
                self._kdtree_float = construct_tree_float(self._data_pts_data_float, self.ndim,
                                                          self.n, self.leafsize)
        else:
            with nogil:
                self._kdtree_double = construct_tree_double(self._data_pts_data_double, self.ndim,
                                                            self.n, self.leafsize)


    def query(KDTree self, np.ndarray query_pts not None, k=1, eps=0,
              distance_upper_bound=None, sqr_dists=False, mask=None):
        """Query the kd-tree for nearest neighbors

        :Parameters:
        query_pts : numpy array
            Query points with shape (m, dims)
        k : int
            The number of nearest neighbours to return
        eps : non-negative float
            Return approximate nearest neighbours; the k-th returned value
            is guaranteed to be no further than (1 + eps) times the distance
            to the real k-th nearest neighbour
        distance_upper_bound : non-negative float
            Return only neighbors within this distance.
            This is used to prune tree searches.
        sqr_dists : bool, optional
            Internally pykdtree works with squared distances.
            Determines if the squared or Euclidean distances are returned.
        mask : numpy array, optional
            Array of booleans where neighbors are considered invalid and
            should not be returned. A mask value of True represents an
            invalid pixel. Mask should have shape (n,) to match data points.
            By default all points are considered valid.

        """

        # Check arguments
        if k < 1:
            raise ValueError('Number of neighbours must be greater than zero')
        elif eps < 0:
            raise ValueError('eps must be non-negative')
        elif distance_upper_bound is not None:
            if distance_upper_bound < 0:
                raise ValueError('distance_upper_bound must be non negative')

        # Check dimensions
        if query_pts.ndim == 1:
            q_ndim = 1
        else:
            q_ndim = query_pts.shape[1]

        if self.ndim != q_ndim:
            raise ValueError('Data and query points must have same dimensions')

        if self.data_pts.dtype == np.float32 and query_pts.dtype != np.float32:
            raise TypeError('Type mismatch. query points must be of type float32 when data points are of type float32')

        # Get query info
        cdef uint32_t num_qpoints = query_pts.shape[0]
        cdef uint32_t num_n = k
        cdef np.ndarray[uint32_t, ndim=1] closest_idxs = np.empty(num_qpoints * k, dtype=np.uint32)
        cdef np.ndarray[float, ndim=1] closest_dists_float
        cdef np.ndarray[double, ndim=1] closest_dists_double


        # Set up return arrays
        cdef uint32_t *closest_idxs_data = <uint32_t *>closest_idxs.data
        cdef float *closest_dists_data_float
        cdef double *closest_dists_data_double
 
        # Get query points data      
        cdef np.ndarray[float, ndim=1] query_array_float 
        cdef np.ndarray[double, ndim=1] query_array_double 
        cdef float *query_array_data_float 
        cdef double *query_array_data_double
        cdef np.ndarray[np.uint8_t, ndim=1] query_mask
        cdef np.uint8_t *query_mask_data

        if mask is not None and mask.size != self.n:
            raise ValueError('Mask must have the same size as data points')
        elif mask is not None:
            query_mask = np.ascontiguousarray(mask.ravel(), dtype=np.uint8)
            query_mask_data = <uint8_t *>query_mask.data
        else:
            query_mask_data = NULL


        if query_pts.dtype == np.float32 and self.data_pts.dtype == np.float32:
            closest_dists_float = np.empty(num_qpoints * k, dtype=np.float32)
            closest_dists = closest_dists_float
            closest_dists_data_float = <float *>closest_dists_float.data
            query_array_float = np.ascontiguousarray(query_pts.ravel(), dtype=np.float32)
            query_array_data_float = <float *>query_array_float.data
        else:
            closest_dists_double = np.empty(num_qpoints * k, dtype=np.float64)
            closest_dists = closest_dists_double
            closest_dists_data_double = <double *>closest_dists_double.data
            query_array_double = np.ascontiguousarray(query_pts.ravel(), dtype=np.float64)
            query_array_data_double = <double *>query_array_double.data

        # Setup distance_upper_bound
        cdef float dub_float
        cdef double dub_double
        if distance_upper_bound is None:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>np.finfo(np.float32).max
            else:
                dub_double = <double>np.finfo(np.float64).max
        else:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>(distance_upper_bound * distance_upper_bound)
            else:
                dub_double = <double>(distance_upper_bound * distance_upper_bound)

        # Set epsilon
        cdef double epsilon_float = <float>eps
        cdef double epsilon_double = <double>eps

        # Release GIL and query tree
        if self.data_pts.dtype == np.float32:
            with nogil:
                search_tree_float(self._kdtree_float, self._data_pts_data_float, 
                                  query_array_data_float, num_qpoints, num_n, dub_float, epsilon_float,
                                  query_mask_data, closest_idxs_data, closest_dists_data_float)

        else:
            with nogil:
                search_tree_double(self._kdtree_double, self._data_pts_data_double, 
                                  query_array_data_double, num_qpoints, num_n, dub_double, epsilon_double,
                                   query_mask_data, closest_idxs_data, closest_dists_data_double)
        
        # Shape result
        if k > 1:
            closest_dists_res = closest_dists.reshape(num_qpoints, k)
            closest_idxs_res = closest_idxs.reshape(num_qpoints, k)
        else:
            closest_dists_res = closest_dists
            closest_idxs_res = closest_idxs

        if distance_upper_bound is not None: # Mark out of bounds results
            if self.data_pts.dtype == np.float32:
                idx_out = (closest_dists_res >= dub_float)
            else:
                idx_out = (closest_dists_res >= dub_double)

            closest_dists_res[idx_out] = np.Inf
            closest_idxs_res[idx_out] = self.n

        if not sqr_dists: # Return actual cartesian distances
            closest_dists_res = np.sqrt(closest_dists_res)

        return closest_dists_res, closest_idxs_res

    def __dealloc__(KDTree self):
        if self._kdtree_float != NULL:
            delete_tree_float(self._kdtree_float)
        elif self._kdtree_double != NULL:
            delete_tree_double(self._kdtree_double)
