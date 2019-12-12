import pyredner_tensorflow as pyredner
import tensorflow as tf
from typing import Optional

class Object:
    """
        Object combines geometry, material, and lighting information
        and aggregate them in a single class. This is a convinent class
        for constructing redner scenes.

        redner supports only triangle meshes for now. It stores a pool of
        vertices and access the pool using integer index. Some times the
        two vertices can have the same 3D position but different texture
        coordinates, because UV mapping creates seams and need to duplicate
        vertices. In this can we can use an additional "uv_indices" array
        to access the uv pool.

        Args
        ====
        vertices: tf.Tensor
            3D position of vertices
            float32 tensor with size num_vertices x 3
        indices: tf.Tensor
            vertex indices of triangle faces.
            int32 tensor with size num_triangles x 3
        material: pyredner.Material

        light_intensity: Optional[tf.Tensor]
            make this object an area light
            float32 tensor with size 3
        light_two_sided: boolean
            Does the light emit from two sides of the shape?
        uvs: Optional[tf.Tensor]:
            optional texture coordinates.
            float32 tensor with size num_uvs x 2
            doesn't need to be the same size with vertices if uv_indices is None
        normals: Optional[tf.Tensor]
            shading normal
            float32 tensor with size num_normals x 3
            doesn't need to be the same size with vertices if normal_indices is None
        uv_indices: Optional[tf.Tensor]
            overrides indices when accessing uv coordinates
            int32 tensor with size num_uvs x 2
        normal_indices: Optional[tf.Tensor]
            overrides indices when accessing shading normals
            int32 tensor with size num_normals x 2
        colors: Optional[tf.Tensor]
            optional per-vertex color
            float32 tensor with size num_vertices x 3
    """
    def __init__(self,
                 vertices: tf.Tensor,
                 indices: tf.Tensor,
                 material: pyredner.Material,
                 light_intensity: Optional[tf.Tensor] = None,
                 light_two_sided: bool = False,
                 uvs: Optional[tf.Tensor] = None,
                 normals: Optional[tf.Tensor] = None,
                 uv_indices: Optional[tf.Tensor] = None,
                 normal_indices: Optional[tf.Tensor] = None,
                 colors: Optional[tf.Tensor] = None):
        assert(vertices.dtype == tf.float32)
        assert(indices.dtype == tf.int32)
        if uvs is not None:
            assert(uvs.dtype == tf.float32)
        if normals is not None:
            assert(normals.dtype == tf.float32)
        if uv_indices is not None:
            assert(uv_indices.dtype == tf.int32)
        if normal_indices is not None:
            assert(normal_indices.dtype == tf.int32)
        if colors is not None:
            assert(colors.dtype == tf.float32)

        self.vertices = vertices
        self.indices = indices
        self.uvs = uvs
        self.normals = normals
        self.uv_indices = uv_indices
        self.normal_indices = normal_indices
        self.colors = colors
        self.material = material
        self.light_intensity = light_intensity
        self.light_two_sided = light_two_sided
