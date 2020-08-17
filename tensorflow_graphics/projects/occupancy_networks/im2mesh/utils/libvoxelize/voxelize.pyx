cimport cython
from libc.math cimport floor, ceil
from cython.view cimport array as cvarray

cdef extern from "tribox2.h":
    int triBoxOverlap(float boxcenter[3], float boxhalfsize[3],
                      float tri0[3], float tri1[3], float tri2[3])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int voxelize_mesh_(bint[:, :, :] occ, float[:, :, ::1] faces):
    assert(faces.shape[1] == 3)
    assert(faces.shape[2] == 3)

    n_faces = faces.shape[0]
    cdef int i
    for i in range(n_faces):
        voxelize_triangle_(occ, faces[i])


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef int voxelize_triangle_(bint[:, :, :] occupancies, float[:, ::1] triverts):
    cdef int bbox_min[3]
    cdef int bbox_max[3]
    cdef int i, j, k
    cdef float boxhalfsize[3]
    cdef float boxcenter[3]
    cdef bint intersection

    boxhalfsize[:] = (0.5, 0.5, 0.5)

    for i in range(3):
        bbox_min[i] = <int> (
            min(triverts[0, i], triverts[1, i], triverts[2, i])
        )
        bbox_min[i] = min(max(bbox_min[i], 0), occupancies.shape[i] - 1)

    for i in range(3):
        bbox_max[i] = <int> (
            max(triverts[0, i], triverts[1, i], triverts[2, i])
        )
        bbox_max[i] = min(max(bbox_max[i], 0), occupancies.shape[i] - 1)

    for i in range(bbox_min[0], bbox_max[0] + 1): 
        for j in range(bbox_min[1], bbox_max[1] + 1): 
            for k in range(bbox_min[2], bbox_max[2] + 1):
                boxcenter[:] = (i + 0.5, j + 0.5, k + 0.5)
                intersection = triBoxOverlap(&boxcenter[0], &boxhalfsize[0],
                                             &triverts[0, 0], &triverts[1, 0], &triverts[2, 0])
                occupancies[i, j, k] |= intersection


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef int test_triangle_aabb(float[::1] boxcenter, float[::1] boxhalfsize, float[:, ::1] triverts):
    assert(boxcenter.shape[0] == 3)
    assert(boxhalfsize.shape[0] == 3)
    assert(triverts.shape[0] == triverts.shape[1] == 3)
    
    # print(triverts)
    # Call functions
    cdef int result = triBoxOverlap(&boxcenter[0], &boxhalfsize[0],
                                    &triverts[0, 0], &triverts[1, 0], &triverts[2, 0])
    return result
