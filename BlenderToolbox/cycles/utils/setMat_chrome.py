import bpy
from include import *

def setMat_chrome(mesh, roughness):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # construct car paint node
    LW = tree.nodes.new('ShaderNodeLayerWeight')
    LW.inputs[0].default_value = 0.7
    CR = tree.nodes.new('ShaderNodeValToRGB')
    CR.color_ramp.elements[0].position = 0.9
    CR.color_ramp.elements[1].position = 1
    CR.color_ramp.elements[0].color = (1,1,1,1)
    CR.color_ramp.elements[1].color = (0,0,0,1)
    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    GLO.inputs[1].default_value = roughness
    GLO.location.x -= 200
    
    tree.links.new(LW.outputs[1], CR.inputs['Fac'])
    tree.links.new(CR.outputs['Color'], GLO.inputs['Color'])
    tree.links.new(GLO.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
