import bpy
from include import *

def setMat_ceramic(mesh, meshC, subC):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    C1Node = initColorNode(tree, meshC)
    C2Node = initColorNode(tree, subC, [200, 400], [200, 200])

    DIF = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(C1Node.outputs['Color'], DIF.inputs['Color'])
    SUB1 = tree.nodes.new('ShaderNodeSubsurfaceScattering')
    SUB1.inputs['Scale'].default_value = 0.3
    tree.links.new(C2Node.outputs['Color'], SUB1.inputs['Color'])
    
    MIX2 = tree.nodes.new('ShaderNodeMixShader')
    MIX2.inputs['Fac'].default_value = 0.35
    tree.links.new(DIF.outputs['BSDF'], MIX2.inputs[1])
    tree.links.new(SUB1.outputs[0], MIX2.inputs[2])

    LW = tree.nodes.new('ShaderNodeLayerWeight')
    LW.inputs['Blend'].default_value = 0.35
    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    
    MIX3 = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(LW.outputs[0], MIX3.inputs['Fac'])
    tree.links.new(MIX2.outputs[0], MIX3.inputs[1])
    tree.links.new(GLO.outputs['BSDF'], MIX3.inputs[2])

    tree.links.new(MIX3.outputs[0], tree.nodes['Material Output'].inputs['Surface'])

	