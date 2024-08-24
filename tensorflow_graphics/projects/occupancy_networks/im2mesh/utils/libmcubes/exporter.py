
import numpy as np


def export_obj(vertices, triangles, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    
    with open(filename, 'w') as fh:
        
        for v in vertices:
            fh.write("v {} {} {}\n".format(*v))
            
        for f in triangles:
            fh.write("f {} {} {}\n".format(*(f + 1)))


def export_off(vertices, triangles, filename):
    """
    Exports a mesh in the (.off) format.
    """
    
    with open(filename, 'w') as fh:
        fh.write('OFF\n')
        fh.write('{} {} 0\n'.format(len(vertices), len(triangles)))

        for v in vertices:
            fh.write("{} {} {}\n".format(*v))
            
        for f in triangles:
            fh.write("3 {} {} {}\n".format(*f))


def export_mesh(vertices, triangles, filename, mesh_name="mcubes_mesh"):
    """
    Exports a mesh in the COLLADA (.dae) format.
    
    Needs PyCollada (https://github.com/pycollada/pycollada).
    """
    
    import collada
    
    mesh = collada.Collada()
    
    vert_src = collada.source.FloatSource("verts-array", vertices, ('X','Y','Z'))
    geom = collada.geometry.Geometry(mesh, "geometry0", mesh_name, [vert_src])
    
    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', "#verts-array")
    
    triset = geom.createTriangleSet(np.copy(triangles), input_list, "")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)
    
    geomnode = collada.scene.GeometryNode(geom, [])
    node = collada.scene.Node(mesh_name, children=[geomnode])
    
    myscene = collada.scene.Scene("mcubes_scene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene
    
    mesh.write(filename)
