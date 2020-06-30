import bpy
import numpy as np

def setMat_monotone(mesh, meshColor, CList, silhouetteColor, shadowSize):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    numColor = len(CList) # numColor >= 3

    # set principled	
    principleNode = tree.nodes["Principled BSDF"]
    principleNode.inputs['Roughness'].default_value = 1.0
    principleNode.inputs['Sheen Tint'].default_value = 0
    principleNode.inputs['Specular'].default_value = 0.0

    # init level
    initRGB = tree.nodes.new('ShaderNodeHueSaturation')
    initRGB.inputs['Color'].default_value = (.5,.5,.5,1)
    initRGB.inputs['Hue'].default_value = 0
    initRGB.inputs['Saturation'].default_value = 0
    initRGB.inputs['Value'].default_value = CList[0].brightness
    initDiff = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(initRGB.outputs['Color'], initDiff.inputs['Color'])

    # init array
    RGBList = [None] * (numColor - 1)
    RampList = [None] * (numColor - 1)
    MixList = [None] * (numColor - 1)
    DiffList = [None] * (numColor - 1)
    for ii in range(numColor-1):
        # RGB node
        RGBList[ii] = tree.nodes.new('ShaderNodeHueSaturation')
        RGBList[ii].inputs['Color'].default_value = (.5,.5,.5,1)
        RGBList[ii].inputs['Hue'].default_value = 0
        RGBList[ii].inputs['Saturation'].default_value = 0
        RGBList[ii].inputs['Value'].default_value = CList[ii+1].brightness
        # Diffuse after RGB
        DiffList[ii] = tree.nodes.new('ShaderNodeBsdfDiffuse')
        # Color Ramp
        RampList[ii] = tree.nodes.new('ShaderNodeValToRGB')
        RampList[ii].color_ramp.interpolation = 'EASE'
        RampList[ii].color_ramp.elements.new(0.5)
        RampList[ii].color_ramp.elements[1].position = CList[ii+1].rampElement1_pos
        RampList[ii].color_ramp.elements[1].color = (0,0,0,1)
        RampList[ii].color_ramp.elements[2].position = CList[ii+1].rampElement2_pos
        # Mix shader
        MixList[ii] = tree.nodes.new('ShaderNodeMixShader')
        # Link shaders
        if ii > 0 and ii < (numColor-1):
            tree.links.new(MixList[ii-1].outputs['Shader'], MixList[ii].inputs[1])
        tree.links.new(RampList[ii].outputs['Color'], MixList[ii].inputs[0])
        tree.links.new(RGBList[ii].outputs['Color'], DiffList[ii].inputs['Color'])
        tree.links.new(DiffList[ii].outputs['BSDF'], MixList[ii].inputs[2])
        # set node location

    # color of the mesh
    mainColor_HSV = tree.nodes.new('ShaderNodeHueSaturation')
    mainColor_HSV.inputs['Color'].default_value = meshColor.RGBA
    mainColor_HSV.inputs['Hue'].default_value = meshColor.H
    mainColor_HSV.inputs['Saturation'].default_value = meshColor.S
    mainColor_HSV.inputs['Value'].default_value = meshColor.V

    # main color BC
    mainColor = tree.nodes.new('ShaderNodeBrightContrast')
    mainColor.inputs['Bright'].default_value = meshColor.B
    mainColor.inputs['Contrast'].default_value = meshColor.C
    tree.links.new(mainColor_HSV.outputs['Color'], mainColor.inputs['Color'])

    # initial and end links
    addShader = tree.nodes.new('ShaderNodeAddShader')
    tree.links.new(initDiff.outputs['BSDF'], MixList[0].inputs[1])
    tree.links.new(MixList[-1].outputs['Shader'], addShader.inputs[0])
    tree.links.new(mainColor.outputs['Color'], principleNode.inputs['Base Color'])
    tree.links.new(principleNode.outputs['BSDF'], addShader.inputs[1])

    # add silhouette 
    mixEnd = tree.nodes.new('ShaderNodeMixShader')
    tree.links.new(mixEnd.outputs['Shader'], tree.nodes['Material Output'].inputs['Surface'])

    edgeShadow_HSV = tree.nodes.new('ShaderNodeHueSaturation')
    edgeShadow_HSV.inputs['Color'].default_value = silhouetteColor.RGBA
    edgeShadow_HSV.inputs['Hue'].default_value = silhouetteColor.H
    edgeShadow_HSV.inputs['Saturation'].default_value = silhouetteColor.S
    edgeShadow_HSV.inputs['Value'].default_value = silhouetteColor.V

    edgeShadow = tree.nodes.new('ShaderNodeBrightContrast')
    edgeShadow.inputs['Bright'].default_value = silhouetteColor.B
    edgeShadow.inputs['Contrast'].default_value = silhouetteColor.C
    
    tree.links.new(edgeShadow_HSV.outputs['Color'], edgeShadow.inputs['Color'])

    diffEnd = tree.nodes.new('ShaderNodeBsdfDiffuse')
    tree.links.new(edgeShadow.outputs['Color'], diffEnd.inputs['Color'])
    tree.links.new(diffEnd.outputs['BSDF'], mixEnd.inputs[2])

    fresnelEnd = tree.nodes.new('ShaderNodeFresnel')
    RampEnd = tree.nodes.new('ShaderNodeValToRGB')
    RampEnd.color_ramp.elements[1].position = shadowSize
    tree.links.new(fresnelEnd.outputs[0], RampEnd.inputs['Fac'])
    tree.links.new(RampEnd.outputs['Color'], mixEnd.inputs[0])
    tree.links.new(addShader.outputs[0], mixEnd.inputs[1])

    # add normal to the color
    fresnelNode = tree.nodes.new('ShaderNodeFresnel')
    textureNode = tree.nodes.new('ShaderNodeTexCoord')
    tree.links.new(textureNode.outputs['Normal'], fresnelNode.inputs['Normal'])
    for ii in range(len(RampList)):
        tree.links.new(fresnelNode.outputs[0], RampList[ii].inputs['Fac'])

    # set node location
    for node in tree.nodes:
        node.location.x = 0
        node.location.y = 0
    yLoc = 0
    xLoc = -400
    RampEnd.location.x = xLoc
    RampEnd.location.y = -300
    for node in RampList:
        node.location.x = xLoc
        node.location.y = yLoc
        yLoc += 300
    yLoc = 0
    xLoc = -600
    initRGB.location.x = xLoc
    initRGB.location.y = -200
    for node in RGBList:
        node.location.x = xLoc
        node.location.y = yLoc
        yLoc += 200

    mainColor.location.x = -800
    mainColor.location.y = 0
    mainColor_HSV.location.x = -1000
    mainColor_HSV.location.y = 0
    edgeShadow.location.x = -800
    edgeShadow.location.y = 200
    edgeShadow_HSV.location.x = -1000
    edgeShadow_HSV.location.y = 200
