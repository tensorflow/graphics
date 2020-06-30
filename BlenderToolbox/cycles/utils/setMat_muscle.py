import bpy

# follow the instruction by Ned Poreyra
def setMat_muscle(mesh, meshColor, fiberShape, bumpStrength = 0.4, wrinkleness = 0.03, maxBrightness = 0.85, minBrightness = 0.1):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # set principled BSDF
    PRIN = tree.nodes["Principled BSDF"]
    PRIN.inputs['Roughness'].default_value = 0.5
    PRIN.inputs['Clearcoat Roughness'].default_value = 0.2

    # initialize required nodes
    BUMP = tree.nodes.new('ShaderNodeBump')
    BUMP.invert = True
    BUMP.inputs[0].default_value = bumpStrength # this controls bump strength
    BUMP.inputs[1].default_value = wrinkleness # (hard to describe) it controls the wrinkle effect
    BUMP.location.x -= 300

    VORONOI = tree.nodes.new('ShaderNodeTexVoronoi')

    MAP = tree.nodes.new('ShaderNodeMapping')
    MAP.vector_type = "POINT"
    MAP.scale[0] = fiberShape[0] # this controls the shape of the fiber
    MAP.scale[1] = fiberShape[1] # this controls the shape of the fiber
    MAP.scale[2] = fiberShape[2] # this controls the shape of the fiber
    MAP.location.x -= 700

    COORD = tree.nodes.new('ShaderNodeTexCoord')

    RAMP_V = tree.nodes.new('ShaderNodeValToRGB')
    RAMP_V.color_ramp.elements[1].position = maxBrightness
    RAMP_V.location.x -= 600
    RAMP_V.location.y -= 300

    RAMP_N = tree.nodes.new('ShaderNodeValToRGB')
    RAMP_N.color_ramp.elements[0].position = minBrightness
    RAMP_N.location.x -= 300
    RAMP_N.location.y -= 300

    NOISE = tree.nodes.new('ShaderNodeTexNoise')
    NOISE.inputs["Scale"].default_value = 15
    NOISE.inputs["Detail"].default_value = 2

    SOFT = tree.nodes.new('ShaderNodeMixRGB')
    SOFT.blend_type = 'SOFT_LIGHT'
    SOFT.inputs[0].default_value = 1.0

    MIX = tree.nodes.new('ShaderNodeMixRGB')
    MIX.inputs[1].default_value = (80/255., 24/255., 16/255., 1)
    MIX.inputs[2].default_value = (38/255., 5/255., 2/255., 1)

    HSV = tree.nodes.new('ShaderNodeHueSaturation')
    HSV.inputs['Saturation'].default_value = meshColor.S
    HSV.inputs['Value'].default_value = meshColor.V
    HSV.inputs['Hue'].default_value = meshColor.H
    HSV.location.x -= 300
    HSV.location.y += 200

    BC = tree.nodes.new('ShaderNodeBrightContrast')
    BC.inputs['Bright'].default_value = meshColor.B
    BC.inputs['Contrast'].default_value = meshColor.C
    BC.location.x -= 600
    BC.location.y += 200

    # create all links
    tree.links.new(HSV.outputs['Color'], BC.inputs['Color'])
    tree.links.new(MIX.outputs[0], HSV.inputs["Color"])
    tree.links.new(BC.outputs['Color'], PRIN.inputs["Base Color"])
    tree.links.new(BUMP.outputs[0], PRIN.inputs['Normal'])
    tree.links.new(RAMP_V.outputs[0], BUMP.inputs[2])
    tree.links.new(RAMP_V.outputs[0], MIX.inputs[0])
    tree.links.new(SOFT.outputs[0], RAMP_V.inputs[0])
    tree.links.new(VORONOI.outputs[1], SOFT.inputs[1])
    tree.links.new(MAP.outputs[0], VORONOI.inputs[0])
    tree.links.new(MAP.outputs[0], NOISE.inputs[0])
    tree.links.new(NOISE.outputs[1], RAMP_N.inputs[0])
    tree.links.new(RAMP_N.outputs[0], SOFT.inputs[2])
    tree.links.new(COORD.outputs[0], MAP.inputs[0])



