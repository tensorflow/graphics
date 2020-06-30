import bpy

def setMat_VColorAO(mesh, meshVColor, AOPercent):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # read vertex attribute
    tree.nodes.new('ShaderNodeAttribute')
    tree.nodes[-1].attribute_name = "Col"
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    tree.links.new(tree.nodes["Attribute"].outputs['Color'], HSVNode.inputs['Color'])
    HSVNode.inputs['Saturation'].default_value = meshVColor.S
    HSVNode.inputs['Value'].default_value = meshVColor.V
    HSVNode.inputs['Hue'].default_value = meshVColor.H
    HSVNode.location.x -= 200

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = meshVColor.B
    BCNode.inputs['Contrast'].default_value = meshVColor.C
    BCNode.location.x -= 400

    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 1.0
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    # add Ambient Occlusion
    tree.nodes.new('ShaderNodeAmbientOcclusion')
    tree.nodes.new('ShaderNodeGamma')
    tree.nodes.new('ShaderNodeMixRGB')
    tree.nodes["Mix"].blend_type = 'MULTIPLY'
    tree.nodes["Gamma"].inputs["Gamma"].default_value = 1.5
    tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0

    # link AO node
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], tree.nodes['Mix'].inputs['Color1'])
    tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
    tree.links.new(tree.nodes["Gamma"].outputs['Color'], tree.nodes['Mix'].inputs['Color2'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])

    DIF = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(tree.nodes["Mix"].outputs[0], DIF.inputs[0])

    MIX = tree.nodes.new('ShaderNodeMixShader')
    MIX.inputs[0].default_value = AOPercent
    MIX.location.x -= 200
    MIX.location.y -= 200
    tree.links.new(tree.nodes['Principled BSDF'].outputs[0], MIX.inputs[1])
    tree.links.new(DIF.outputs[0], MIX.inputs[2])
    tree.links.new(MIX.outputs[0], tree.nodes['Material Output'].inputs['Surface'])

    