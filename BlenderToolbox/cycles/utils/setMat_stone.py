import bpy
from include import *

def setMat_stone(mesh, meshColor, noiseScale, distortion, AOStrength):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    BC = initColorNode(tree, meshColor)

    TC = tree.nodes.new('ShaderNodeTexCoord')
    TN = tree.nodes.new('ShaderNodeTexNoise')
    TN.inputs['Scale'].default_value = noiseScale
    TN.inputs['Distortion'].default_value = distortion
    TN.location.x -= 200
    TN.location.y -= 200

    tree.links.new(TC.outputs[0], TN.inputs[0])
    RGB2BW = tree.nodes.new('ShaderNodeRGBToBW')
    tree.links.new(TN.outputs[0], RGB2BW.inputs[0])
    POW = tree.nodes.new('ShaderNodeMath')
    POW.operation = 'POWER'
    tree.links.new(RGB2BW.outputs[0], POW.inputs['Value'])
    MUL = tree.nodes.new('ShaderNodeMath')
    MUL.operation = 'MULTIPLY'
    tree.links.new(POW.outputs[0], MUL.inputs['Value'])
    BP = tree.nodes.new('ShaderNodeBump')
    tree.links.new(MUL.outputs[0], BP.inputs['Height'])
    tree.links.new(MUL.outputs[0], BP.inputs['Normal'])


    DIF1 = tree.nodes.new('ShaderNodeBsdfDiffuse')
    PRIN = tree.nodes["Principled BSDF"]
    tree.links.new(BC.outputs['Color'], DIF1.inputs['Color'])
    tree.links.new(BC.outputs['Color'], PRIN.inputs['Base Color'])
    tree.links.new(BP.outputs[0], DIF1.inputs['Normal'])

    MIX = tree.nodes.new('ShaderNodeMixShader')
    MIX.inputs['Fac'].default_value = 0.5
    tree.links.new(DIF1.outputs[0], MIX.inputs[1])
    tree.links.new(PRIN.outputs[0], MIX.inputs[2])

    AO = tree.nodes.new('ShaderNodeAmbientOcclusion')
    GM = tree.nodes.new('ShaderNodeGamma')
    MIX2 = tree.nodes.new('ShaderNodeMixRGB')
    MIX2.blend_type = 'MULTIPLY'
    GM.inputs["Gamma"].default_value = AOStrength
    GM.location.x -= 600
    AO.inputs["Distance"].default_value = 10.0

    tree.links.new(BC.outputs['Color'], AO.inputs['Color'])
    tree.links.new(AO.outputs['Color'], MIX2.inputs['Color1'])
    tree.links.new(AO.outputs['AO'], GM.inputs['Color'])
    tree.links.new(GM.outputs['Color'], MIX2.inputs['Color2'])
    tree.links.new(MIX2.outputs['Color'], PRIN.inputs['Base Color'])

    tree.links.new(MIX.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
