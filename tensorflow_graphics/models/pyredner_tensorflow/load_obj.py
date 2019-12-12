import tensorflow as tf
import re
import pyredner_tensorflow as pyredner
import os

class WavefrontMaterial:
    def __init__(self):
        self.name = ""
        self.Kd = (0.0, 0.0, 0.0)
        self.Ks = (0.0, 0.0, 0.0)
        self.Ns = 0.0
        self.Ke = (0.0, 0.0, 0.0)
        self.map_Kd = None
        self.map_Ks = None
        self.map_Ns = None

class TriangleMesh:
    def __init__(self,
                 indices,
                 uv_indices,
                 normal_indices,
                 vertices,
                 uvs,
                 normals):
        self.vertices = vertices
        self.indices = indices
        self.uv_indices = uv_indices
        self.normal_indices = normal_indices
        self.uvs = uvs
        self.normals = normals

def load_mtl(filename):
    mtllib = {}
    current_mtl = WavefrontMaterial()
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            splitted = re.split('\ +', line)
            if splitted[0] == 'newmtl':
                if current_mtl.name != "":
                    mtllib[current_mtl.name] = current_mtl
                current_mtl = WavefrontMaterial()
                current_mtl.name = splitted[1]
            elif splitted[0] == 'Kd':
                current_mtl.Kd = (float(splitted[1]), float(splitted[2]), float(splitted[3]))
            elif splitted[0] == 'Ks':
                current_mtl.Ks = (float(splitted[1]), float(splitted[2]), float(splitted[3]))
            elif splitted[0] == 'Ns':
                current_mtl.Ns = float(splitted[1])
            elif splitted[0] == 'Ke':
                current_mtl.Ke = (float(splitted[1]), float(splitted[2]), float(splitted[3]))
            elif splitted[0] == 'map_Kd':
                current_mtl.map_Kd = splitted[1]
            elif splitted[0] == 'map_Ks':
                current_mtl.map_Ks = splitted[1]
            elif splitted[0] == 'map_Ns':
                current_mtl.map_Ns = splitted[1]
    if current_mtl.name != "":
        mtllib[current_mtl.name] = current_mtl
    return mtllib

def load_obj(filename: str,
             obj_group: bool = True,
             flip_tex_coords: bool = True,
             use_common_indices: bool = False,
             return_objects: bool = False):
    """
        Load from a Wavefront obj file as PyTorch tensors.

        Args
        ====
        obj_group: bool
            split the meshes based on materials
        flip_tex_coords: bool
            flip the v coordinate of uv by applying v' = 1 - v
        use_common_indices: bool
            Use the same indices for position, uvs, normals.
            Not recommended since texture seams in the objects sharing
            the same positions would cause the optimization to "tear" the object
        return_objects: bool
            Output list of Object instead.
            If there is no corresponding material for a shape, assign a grey material.

        Returns
        =======
        if return_objects == True, return a list of Object
        if return_objects == False, return (material_map, mesh_list, light_map),
        material_map -> Map[mtl_name, WavefrontMaterial]
        mesh_list -> List[TriangleMesh]
        light_map -> Map[mtl_name, torch.Tensor]
    """
    vertices_pool = []
    uvs_pool = []
    normals_pool = []
    indices = []
    uv_indices = []
    normal_indices = []
    vertices = []
    uvs = []
    normals = []
    vertices_map = {}
    uvs_map = {}
    normals_map = {}
    material_map = {}
    current_mtllib = {}
    current_material_name = None

    def create_mesh(indices,
                    uv_indices,
                    normal_indices,
                    vertices,
                    uvs,
                    normals):
        indices = tf.constant(indices, dtype = tf.int32)
        if len(uv_indices) == 0:
            uv_indices = None
        else:
            uv_indices = tf.constant(uv_indices, dtype = tf.int32)
        if len(normal_indices) == 0:
            normal_indices = None
        else:
            normal_indices = tf.constant(normal_indices, dtype = tf.int32)
        vertices = tf.constant(vertices)
        if len(uvs) == 0:
            uvs = None
        else:
            uvs = tf.constant(uvs)
        if len(normals) == 0:
            normals = None
        else:
            normals = tf.constant(normals)
        return TriangleMesh(indices,
                            uv_indices,
                            normal_indices,
                            vertices,
                            uvs,
                            normals)

    mesh_list = []
    light_map = {}

    with open(filename, 'r') as f:
        d = os.path.dirname(filename)
        cwd = os.getcwd()
        if d != '':
            os.chdir(d)
        for line in f:
            line = line.strip()
            splitted = re.split('\ +', line)
            if splitted[0] == 'mtllib':
                current_mtllib = load_mtl(splitted[1])
            elif splitted[0] == 'usemtl':
                if len(indices) > 0 and obj_group is True:
                    # Flush
                    mesh_list.append((current_material_name,
                        create_mesh(indices, uv_indices, normal_indices,
                                    vertices, uvs, normals)))
                    indices = []
                    uv_indices = []
                    normal_indices = []
                    vertices = []
                    normals = []
                    uvs = []
                    vertices_map = {}
                    uvs_map = {}
                    normals_map = {}

                mtl_name = splitted[1]
                current_material_name = mtl_name
                if mtl_name not in material_map:
                    m = current_mtllib[mtl_name]
                    if m.map_Kd is None:
                        diffuse_reflectance = tf.constant(m.Kd,
                            dtype = tf.float32)
                    else:
                        diffuse_reflectance = pyredner.imread(m.map_Kd)
                    if m.map_Ks is None:
                        specular_reflectance = tf.constant(m.Ks,
                            dtype = tf.float32)
                    else:
                        specular_reflectance = pyredner.imread(m.map_Ks)
                    if m.map_Ns is None:
                        roughness = tf.constant([2.0 / (m.Ns + 2.0)],
                            dtype = tf.float32)
                    else:
                        roughness = 2.0 / (pyredner.imread(m.map_Ks) + 2.0)
                    if m.Ke != (0.0, 0.0, 0.0):
                        light_map[mtl_name] = tf.constant(m.Ke, dtype = tf.float32)
                    material_map[mtl_name] = pyredner.Material(
                        diffuse_reflectance, specular_reflectance, roughness)
            elif splitted[0] == 'v':
                vertices_pool.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
            elif splitted[0] == 'vt':
                u = float(splitted[1])
                v = float(splitted[2])
                if flip_tex_coords:
                    v = 1 - v
                uvs_pool.append([u, v])
            elif splitted[0] == 'vn':
                normals_pool.append([float(splitted[1]), float(splitted[2]), float(splitted[3])])
            elif splitted[0] == 'f':
                def num_indices(x):
                    return len(re.split('/', x))
                def get_index(x, i):
                    return int(re.split('/', x)[i])
                def parse_face_index(x, i):
                    f = get_index(x, i)
                    if f > 0:
                        f -= 1
                    return f
                assert(len(splitted) <= 5)
                def get_vertex_id(indices):
                    pi = parse_face_index(indices, 0)
                    uvi = None
                    if (num_indices(indices) > 1 and re.split('/', indices)[1] != ''):
                        uvi = parse_face_index(indices, 1)
                    ni = None
                    if (num_indices(indices) > 2 and re.split('/', indices)[2] != ''):
                        ni = parse_face_index(indices, 2)
                    if use_common_indices:
                        # vertex, uv, normals share the same indexing
                        key = (pi, uvi, ni)
                        if key in vertices_map:
                            vertex_id = vertices_map[key]
                            return vertex_id, vertex_id, vertex_id

                        vertex_id = len(vertices)
                        vertices_map[key] = vertex_id
                        vertices.append(vertices_pool[pi])
                        if uvi is not None:
                            uvs.append(uvs_pool[uvi])
                        if ni is not None:
                            normals.append(normals_pool[ni])
                        return vertex_id, vertex_id, vertex_id
                    else:
                        # vertex, uv, normals use separate indexing
                        vertex_id = None
                        uv_id = None
                        normal_id = None

                        if pi in vertices_map:
                            vertex_id = vertices_map[pi]
                        else:
                            vertex_id = len(vertices)
                            vertices.append(vertices_pool[pi])
                            vertices_map[pi] = vertex_id

                        if uvi is not None:
                            if uvi in uvs_map:
                                uv_id = uvs_map[uvi]
                            else:
                                uv_id = len(uvs)
                                uvs.append(uvs_pool[uvi])
                                uvs_map[uvi] = uv_id

                        if ni is not None:
                            if ni in normals_map:
                                normal_id = normals_map[ni]
                            else:
                                normal_id = len(normals)
                                normals.append(normals_pool[ni])
                                normals_map[ni] = normal_id
                        return vertex_id, uv_id, normal_id

                vid0, uv_id0, n_id0 = get_vertex_id(splitted[1])
                vid1, uv_id1, n_id1 = get_vertex_id(splitted[2])
                vid2, uv_id2, n_id2 = get_vertex_id(splitted[3])

                indices.append([vid0, vid1, vid2])
                if uv_id0 is not None:
                    assert(uv_id1 is not None and uv_id2 is not None)
                    uv_indices.append([uv_id0, uv_id1, uv_id2])
                if n_id0 is not None:
                    assert(n_id1 is not None and n_id2 is not None)
                    normal_indices.append([n_id0, n_id1, n_id2])
                if (len(splitted) == 5):
                    vid3, uv_id3, n_id3 = get_vertex_id(splitted[4])
                    indices.append([vid0, vid2, vid3])
                    if uv_id0 is not None:
                        assert(uv_id3 is not None)
                        uv_indices.append([uv_id0, uv_id2, uv_id3])
                    if n_id0 is not None:
                        assert(n_id3 is not None)
                        normal_indices.append([n_id0, n_id2, n_id3])
    
    mesh_list.append((current_material_name,
        create_mesh(indices, uv_indices, normal_indices, vertices, uvs, normals)))
    if d != '':
        os.chdir(cwd)

    if return_objects:
        objects = []
        for mtl_name, mesh in mesh_list:
            if mtl_name in material_map:
                m = material_map[mtl_name]
            else:
                m = pyredner.Material(diffuse_reflectance = \
                        tf.constant((0.5, 0.5, 0.5)))
            if mtl_name in light_map:
                l = light_map[mtl_name]
            else:
                l = None
            objects.append(pyredner.Object(\
                vertices = mesh.vertices,
                indices = mesh.indices,
                material = m,
                light_intensity = l,
                uvs = mesh.uvs,
                normals = mesh.normals,
                uv_indices = mesh.uv_indices,
                normal_indices = mesh.normal_indices))
        return objects
    else:
        return material_map, mesh_list, light_map

