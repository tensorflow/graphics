# distutils: language = c++
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np


cdef extern from "Simplify.h":
    cdef struct vec3f:
        double x, y, z

    cdef cppclass SymetricMatrix:
        SymetricMatrix() except +


cdef extern from "Simplify.h" namespace "Simplify":
    cdef struct Triangle:
        int v[3]
        double err[4]
        int deleted, dirty, attr
        vec3f uvs[3]
        int material

    cdef struct Vertex:
        vec3f p
        int tstart, tcount
        SymetricMatrix q
        int border
    
    cdef vector[Triangle] triangles
    cdef vector[Vertex] vertices
    cdef void simplify_mesh(int, double)


cpdef mesh_simplify(double[:, ::1] vertices_in, long[:, ::1] triangles_in,
                        int f_target, double agressiveness=7.) except +:
    vertices.clear()
    triangles.clear()

    # Read in vertices and triangles
    cdef Vertex v
    for iv in range(vertices_in.shape[0]):
        v = Vertex()
        v.p.x = vertices_in[iv, 0]
        v.p.y = vertices_in[iv, 1]
        v.p.z = vertices_in[iv, 2]
        vertices.push_back(v)

    cdef Triangle t
    for it in range(triangles_in.shape[0]):
        t = Triangle()
        t.v[0] = triangles_in[it, 0]
        t.v[1] = triangles_in[it, 1]
        t.v[2] = triangles_in[it, 2]
        triangles.push_back(t)

    # Simplify
    # print('Simplify...')
    simplify_mesh(f_target, agressiveness)

    # Only use triangles that are not deleted
    cdef vector[Triangle] triangles_notdel
    triangles_notdel.reserve(triangles.size())

    for t in triangles:
        if not t.deleted:
            triangles_notdel.push_back(t)

    # Read out triangles
    vertices_out = np.empty((vertices.size(), 3), dtype=np.float64)
    triangles_out = np.empty((triangles_notdel.size(), 3), dtype=np.int64)
    
    cdef double[:, :] vertices_out_view = vertices_out
    cdef long[:, :] triangles_out_view = triangles_out

    for iv in range(vertices.size()):
        vertices_out_view[iv, 0] = vertices[iv].p.x
        vertices_out_view[iv, 1] = vertices[iv].p.y
        vertices_out_view[iv, 2] = vertices[iv].p.z

    for it in range(triangles_notdel.size()):
        triangles_out_view[it, 0] = triangles_notdel[it].v[0]
        triangles_out_view[it, 1] = triangles_notdel[it].v[1]
        triangles_out_view[it, 2] = triangles_notdel[it].v[2]

    # Clear vertices and triangles
    vertices.clear()
    triangles.clear()
    
    return vertices_out, triangles_out