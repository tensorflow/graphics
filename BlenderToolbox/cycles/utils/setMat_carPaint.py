import bpy
from include import *

def setMat_carPaint(mesh, C1, C2):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    C1_BS = initColorNode(tree, C1)
    C2_BS = initColorNode(tree, C2, [200, 400], [200, 200])

    # construct car paint node
    LW = tree.nodes.new('ShaderNodeLayerWeight')
    CR = tree.nodes.new('ShaderNodeValToRGB')
    tree.links.new(LW.outputs['Facing'], CR.inputs['Fac'])

    MIX = tree.nodes.new('ShaderNodeMixRGB')
    tree.links.new(CR.outputs['Color'], MIX.inputs['Fac'])
    tree.links.new(C1_BS.outputs['Color'], MIX.inputs['Color1'])
    tree.links.new(C2_BS.outputs['Color'], MIX.inputs['Color2'])

    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    GLO.inputs['Roughness'].default_value = 0.224
    DIF = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(MIX.outputs['Color'], DIF.inputs['Color'])

    MIXS = tree.nodes.new('ShaderNodeMixShader')
    MIXS.inputs['Fac'].default_value = 0.25
    tree.links.new(DIF.outputs['BSDF'], MIXS.inputs[1])
    tree.links.new(GLO.outputs['BSDF'], MIXS.inputs[2])

    tree.links.new(MIXS.outputs[0], tree.nodes['Material Output'].inputs['Surface'])