
# distutils: language=c++
import numpy as np
cimport numpy as np
cimport cython
from libcpp.vector cimport vector
from libc.math cimport floor, ceil

cdef class TriangleHash:
    cdef vector[vector[int]] spatial_hash
    cdef int resolution

    def __cinit__(self, double[:, :, :] triangles, int resolution):
        self.spatial_hash.resize(resolution * resolution)
        self.resolution = resolution
        self._build_hash(triangles)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cdef int _build_hash(self, double[:, :, :] triangles):
        assert(triangles.shape[1] == 3)
        assert(triangles.shape[2] == 2)

        cdef int n_tri = triangles.shape[0]
        cdef int bbox_min[2]
        cdef int bbox_max[2]
        
        cdef int i_tri, j, x, y
        cdef int spatial_idx

        for i_tri in range(n_tri):
            # Compute bounding box
            for j in range(2):
                bbox_min[j] = <int> min(
                    triangles[i_tri, 0, j], triangles[i_tri, 1, j], triangles[i_tri, 2, j]
                )
                bbox_max[j] = <int> max(
                    triangles[i_tri, 0, j], triangles[i_tri, 1, j], triangles[i_tri, 2, j]
                )
                bbox_min[j] = min(max(bbox_min[j], 0), self.resolution - 1)
                bbox_max[j] = min(max(bbox_max[j], 0), self.resolution - 1)

            # Find all voxels where bounding box intersects
            for x in range(bbox_min[0], bbox_max[0] + 1):
                for y in range(bbox_min[1], bbox_max[1] + 1):
                    spatial_idx = self.resolution * x + y
                    self.spatial_hash[spatial_idx].push_back(i_tri)

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef query(self, double[:, :] points):
        assert(points.shape[1] == 2)
        cdef int n_points = points.shape[0]

        cdef vector[int] points_indices
        cdef vector[int] tri_indices
        # cdef int[:] points_indices_np
        # cdef int[:] tri_indices_np

        cdef int i_point, k, x, y
        cdef int spatial_idx

        for i_point in range(n_points):
            x = int(points[i_point, 0])
            y = int(points[i_point, 1])
            if not (0 <= x < self.resolution and 0 <= y < self.resolution):
                continue

            spatial_idx = self.resolution * x +  y
            for i_tri in self.spatial_hash[spatial_idx]:
                points_indices.push_back(i_point)
                tri_indices.push_back(i_tri)

        points_indices_np = np.zeros(points_indices.size(), dtype=np.int32)
        tri_indices_np = np.zeros(tri_indices.size(), dtype=np.int32)

        cdef int[:] points_indices_view = points_indices_np
        cdef int[:] tri_indices_view = tri_indices_np

        for k in range(points_indices.size()):
            points_indices_view[k] = points_indices[k]

        for k in range(tri_indices.size()):
            tri_indices_view[k] = tri_indices[k]
            
        return points_indices_np, tri_indices_np
