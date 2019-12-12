import tensorflow as tf
import xml.etree.ElementTree as etree
import numpy as np
import redner
import os
import pyredner_tensorflow as pyredner
import pyredner_tensorflow.transform as transform

def parse_transform(node):
    ret = tf.eye(4)
    for child in node:
        if child.tag == 'matrix':
            value = tf.convert_to_tensor(
                np.reshape(
                    np.fromstring(child.attrib['value'], dtype=np.float32, sep=' '),
                    (4, 4)))
            ret = value @ ret
        elif child.tag == 'translate':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = transform.gen_translate_matrix(tf.constant([x, y, z]))
            ret = value @ ret
        elif child.tag == 'scale':
            x = float(child.attrib['x'])
            y = float(child.attrib['y'])
            z = float(child.attrib['z'])
            value = transform.gen_scale_matrix(tf.constant([x, y, z]))
            ret = value @ ret
    return ret

def parse_vector(str):
    v = np.fromstring(str, dtype=np.float32, sep=',')
    if v.shape[0] != 3:
        v = np.fromstring(str, dtype=np.float32, sep=' ')
    assert(v.ndim == 1)
    return tf.convert_to_tensor(v)

def parse_camera(node):
    fov = tf.constant([45.0])
    position = None
    look_at = None
    up = None
    clip_near = 1e-2
    resolution = [256, 256]
    for child in node:
        if 'name' in child.attrib:
            if child.attrib['name'] == 'fov':
                fov = tf.constant([float(child.attrib['value'])])
            elif child.attrib['name'] == 'toWorld':
                has_lookat = False
                for grandchild in child:
                    if grandchild.tag.lower() == 'lookat':
                        has_lookat = True
                        position = parse_vector(grandchild.attrib['origin'])
                        look_at = parse_vector(grandchild.attrib['target'])
                        up = parse_vector(grandchild.attrib['up'])
                if not has_lookat:
                    print('Unsupported Mitsuba scene format: please use a look at transform')
                    assert(False)
        if child.tag == 'film':
            for grandchild in child:
                if 'name' in grandchild.attrib:
                    if grandchild.attrib['name'] == 'width':
                        resolution[0] = int(grandchild.attrib['value'])
                    elif grandchild.attrib['name'] == 'height':
                        resolution[1] = int(grandchild.attrib['value'])

    return pyredner.Camera(position     = position,
                           look_at      = look_at,
                           up           = up,
                           fov          = fov,
                           clip_near    = clip_near,
                           resolution   = resolution)

def parse_material(node, two_sided = False):
    node_id = None
    if 'id' in node.attrib:
        node_id = node.attrib['id']
    if node.attrib['type'] == 'diffuse':
        diffuse_reflectance = tf.constant([0.5, 0.5, 0.5])
        diffuse_uv_scale = [1.0, 1.0]
        specular_reflectance = tf.constant([0.0, 0.0, 0.0])
        specular_uv_scale = [1.0, 1.0]
        roughness = tf.constant([1.0])

        for child in node:
            if child.attrib['name'] == 'reflectance':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            diffuse_reflectance = pyredner.imread(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'uscale':
                            diffuse_uv_scale[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            diffuse_uv_scale[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'specular':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            specular_reflectance = pyredner.imread(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'uscale':
                            specular_uv_scale[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            specular_uv_scale[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    specular_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'roughness':
                roughness = tf.constant([float(child.attrib['value'])])
        
        diffuse_uv_scale = tf.constant(diffuse_uv_scale)
        specular_uv_scale = tf.constant(specular_uv_scale)

        return (node_id, pyredner.Material(
                diffuse_reflectance = pyredner.Texture(diffuse_reflectance, diffuse_uv_scale),
                specular_reflectance = pyredner.Texture(specular_reflectance, specular_uv_scale),
                roughness = pyredner.Texture(roughness),
                two_sided = two_sided))

    elif node.attrib['type'] == 'roughplastic':
        diffuse_reflectance = tf.constant([0.5, 0.5, 0.5])
        diffuse_uv_scale = [1.0, 1.0]
        specular_reflectance = tf.constant([0.0, 0.0, 0.0])
        specular_uv_scale = [1.0, 1.0]
        roughness = tf.constant([1.0])
        for child in node:
            if child.attrib['name'] == 'diffuseReflectance':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            diffuse_reflectance = pyredner.imread(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'uscale':
                            diffuse_uv_scale[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            diffuse_uv_scale[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    diffuse_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'specularReflectance':
                if child.tag == 'texture':
                    for grandchild in child:
                        if grandchild.attrib['name'] == 'filename':
                            specular_reflectance = pyredner.imread(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'uscale':
                            specular_uv_scale[0] = float(grandchild.attrib['value'])
                        elif grandchild.attrib['name'] == 'vscale':
                            specular_uv_scale[1] = float(grandchild.attrib['value'])
                elif child.tag == 'rgb' or child.tag == 'spectrum':
                    specular_reflectance = parse_vector(child.attrib['value'])
            elif child.attrib['name'] == 'alpha':
                alpha = float(child.attrib['value'])
                roughness = tf.constant([alpha * alpha])
        
        diffuse_uv_scale = tf.constant(diffuse_uv_scale)
        specular_uv_scale = tf.constant(specular_uv_scale)

        return (node_id, pyredner.Material(
                diffuse_reflectance = pyredner.Texture(diffuse_reflectance, diffuse_uv_scale),
                specular_reflectance = pyredner.Texture(specular_reflectance, specular_uv_scale),
                roughness = pyredner.Texture(roughness),
                two_sided = two_sided))
    elif node.attrib['type'] == 'twosided':
        ret = parse_material(node[0], True)
        return (node_id, ret[1])
    else:
        print('Unsupported material type:', node.attrib['type'])
        assert(False)

def parse_shape(node, material_dict, shape_id):
    if node.attrib['type'] == 'obj' or node.attrib['type'] == 'serialized':
        to_world = tf.eye(4)
        serialized_shape_id = 0
        mat_id = -1
        light_intensity = None
        filename = ''
        max_smooth_angle = -1
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'filename':
                    filename = child.attrib['value']
                elif child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
                elif child.attrib['name'] == 'shapeIndex':
                    serialized_shape_id = int(child.attrib['value'])
                elif child.attrib['name'] == 'maxSmoothAngle':
                    max_smooth_angle = float(child.attrib['value'])
            if child.tag == 'ref':
                mat_id = material_dict[child.attrib['id']]
            elif child.tag == 'emitter':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = tf.constant(
                                         [light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]])

        if node.attrib['type'] == 'obj':
            _, mesh_list, _ = pyredner.load_obj(filename)
            # Convert to CPU for rebuild_topology
            vertices = mesh_list[0][1].vertices.cpu()
            indices = mesh_list[0][1].indices.cpu()
            uvs = mesh_list[0][1].uvs
            normals = mesh_list[0][1].normals
            uv_indices = mesh_list[0][1].uv_indices
            normal_indices = mesh_list[0][1].normal_indices
            if uvs is not None:
                uvs = uvs.cpu()
            if normals is not None:
                normals = normals.cpu()
            if uv_indices is not None:
                uv_indices = uv_indices.cpu()
        else:
            assert(node.attrib['type'] == 'serialized')
            mitsuba_tri_mesh = redner.load_serialized(filename, serialized_shape_id)
            vertices = tf.convert_to_tensor(mitsuba_tri_mesh.vertices)
            indices = tf.convert_to_tensor(mitsuba_tri_mesh.indices)
            uvs = tf.convert_to_tensor(mitsuba_tri_mesh.uvs)
            normals = tf.convert_to_tensor(mitsuba_tri_mesh.normals)
            if uvs.shape[0] == 0:
                uvs = None
            if normals.shape[0] == 0:
                normals = None

        # Transform the vertices and normals
        vertices = tf.concat((vertices, tf.ones([vertices.shape[0], 1], dtype=tf.float32)), axis = 1)
        vertices = vertices @ tf.transpose(to_world, [0, 1])
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3]
        if normals is not None:
            normals = normals @ (tf.linalg.inv(tf.transpose(to_world, [0, 1]))[:3, :3])
        assert(vertices is not None)
        assert(indices is not None)
        if max_smooth_angle >= 0:
            if normals is None:
                normals = tf.zeros_like(vertices)
            new_num_vertices = redner.rebuild_topology(\
                redner.float_ptr(pyredner.data_ptr(vertices)),
                redner.int_ptr(pyredner.data_ptr(indices)),
                redner.float_ptr(pyredner.data_ptr(uvs) if uvs is not None else 0),
                redner.float_ptr(pyredner.data_ptr(normals) if normals is not None else 0),
                redner.int_ptr(pyredner.data_ptr(uv_indices) if uv_indices is not None else 0),
                int(vertices.shape[0]),
                int(indices.shape[0]),
                max_smooth_angle)
            print('Rebuilt topology, original vertices size: {}, new vertices size: {}'.format(\
                int(vertices.shape[0]), new_num_vertices))
            vertices.resize_(new_num_vertices, 3)
            if uvs is not None:
                uvs.resize_(new_num_vertices, 2)
            if normals is not None:
                normals.resize_(new_num_vertices, 3)

        lgt = None
        if light_intensity is not None:
            lgt = pyredner.AreaLight(shape_id, light_intensity)

        return pyredner.Shape(vertices=vertices,
                              indices=indices,
                              uvs=uvs,
                              normals=normals,
                              material_id=mat_id), lgt
    elif node.attrib['type'] == 'rectangle':
        indices = tf.constant([[0, 2, 1], [1, 2, 3]],
                               dtype = tf.int32)
        vertices = tf.constant([[-1.0, -1.0, 0.0],
                                 [-1.0,  1.0, 0.0],
                                 [ 1.0, -1.0, 0.0],
                                 [ 1.0,  1.0, 0.0]])
        uvs = None
        normals = None
        to_world = tf.eye(4)
        mat_id = -1
        light_intensity = None
        for child in node:
            if 'name' in child.attrib:
                if child.attrib['name'] == 'toWorld':
                    to_world = parse_transform(child)
            if child.tag == 'ref':
                mat_id = material_dict[child.attrib['id']]
            elif child.tag == 'emitter':
                for grandchild in child:
                    if grandchild.attrib['name'] == 'radiance':
                        light_intensity = parse_vector(grandchild.attrib['value'])
                        if light_intensity.shape[0] == 1:
                            light_intensity = tf.constant(
                                         [light_intensity[0],
                                          light_intensity[0],
                                          light_intensity[0]])
        # Transform the vertices and normals
        vertices = tf.concat((vertices, tf.convert_to_tensor(np.ones(vertices.shape[0], 1), dtype=tf.float32)), axis = 1)
        vertices = vertices @ tf.transpose(to_world, [0, 1])
        vertices = vertices / vertices[:, 3:4]
        vertices = vertices[:, 0:3]
        if normals is not None:
            normals = normals @ (tf.linalg.inv(tf.transpose(to_world, [0, 1]))[:3, :3])
        assert(vertices is not None)
        assert(indices is not None)
        lgt = None
        if light_intensity is not None:
            lgt = pyrender.Light(shape_id, light_intensity)

        
        return pyredner.Shape(vertices=vertices,
                              indices=indices,
                              uvs=uvs,
                              normals=normals,
                              material_id=mat_id), lgt
    else:
        assert(False)

def parse_scene(node):
    cam = None
    resolution = None
    materials = []
    material_dict = {}
    shapes = []
    lights = []
    for child in node:
        if child.tag == 'sensor':
            cam = parse_camera(child)
        elif child.tag == 'bsdf':
            node_id, material = parse_material(child)
            if node_id is not None:
                material_dict[node_id] = len(materials)
                materials.append(material)
        elif child.tag == 'shape':
            shape, light = parse_shape(child, material_dict, len(shapes))
            shapes.append(shape)
            if light is not None:
                lights.append(light)
    return pyredner.Scene(cam, shapes, materials, lights)

def load_mitsuba(filename):
    """
        Load from a Mitsuba scene file as PyTorch tensors.
    """

    tree = etree.parse(filename)
    root = tree.getroot()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(filename))
    ret = parse_scene(root)
    os.chdir(cwd)
    return ret
