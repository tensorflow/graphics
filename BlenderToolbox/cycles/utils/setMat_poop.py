import bpy
from include import *


def setMat_poop(mesh, poopRGB1, poopRGB2, noiseScale, noiseDetail, noiseDistortion, brightness):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    TN = tree.nodes.new('ShaderNodeTexNoise')
    TN.inputs['Scale'].default_value = noiseScale
    TN.inputs['Detail'].default_value = noiseDetail
    TN.inputs['Distortion'].default_value = noiseDistortion
    TN.location.x -= 200

    MUL = tree.nodes.new('ShaderNodeMath')
    MUL.inputs[1].default_value = 3.9
    MUL.operation = 'MULTIPLY'
    tree.links.new(TN.outputs[0], MUL.inputs['Value'])

    CR1 = tree.nodes.new('ShaderNodeValToRGB')
    CR1.color_ramp.elements[0].position = brightness
    CR1.location.x -= 500

    CR2 = tree.nodes.new('ShaderNodeValToRGB')
    CR2.color_ramp.elements[0].position = 0.605
    CR2.color_ramp.elements[0].color = poopRGB1
    CR2.color_ramp.elements[1].color = poopRGB2
    CR2.location.x -= 800

    PRIN = tree.nodes["Principled BSDF"]
    PRIN.inputs["Specular"].default_value = 0
    tree.links.new(CR2.outputs[0], PRIN.inputs[0])

    GLO = tree.nodes.new('ShaderNodeBsdfGlossy')
    GLO.inputs['Color'].default_value = poopRGB1
    GLO.inputs['Roughness'].default_value = 0.117

    MIX = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(CR1.outputs[0], MIX.inputs[0])
    tree.links.new(PRIN.outputs[0], MIX.inputs[1])
    tree.links.new(GLO.outputs[0], MIX.inputs[2])

    tree.links.new(MIX.outputs[0], tree.nodes['Material Output'].inputs['Surface'])
    tree.links.new(MUL.outputs[0], tree.nodes['Material Output'].inputs['Displacement'])