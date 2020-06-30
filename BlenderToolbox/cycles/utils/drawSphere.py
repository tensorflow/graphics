import bpy
from include import *

def drawSphere(ptSize, ptColor, ptLoc = (1e10,1e10,1e10)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius = ptSize)
    sphere = bpy.context.object
    sphere.location = ptLoc
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new('sphere_mat')
    sphere.data.materials.append(mat)
    mat.use_nodes = True
    tree = mat.node_tree

    BCNode = initColorNode(tree, ptColor)

    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 1.0
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
    return sphere