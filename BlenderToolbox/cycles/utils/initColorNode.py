import bpy

def initColorNode(tree, color, xloc = [200,400], yloc = [0,0]):
    HSV = tree.nodes.new('ShaderNodeHueSaturation')
    HSV.inputs['Color'].default_value = color.RGBA
    HSV.inputs['Saturation'].default_value = color.S
    HSV.inputs['Value'].default_value = color.V
    HSV.inputs['Hue'].default_value = color.H
    HSV.location.x -= xloc[0]
    HSV.location.y -= yloc[0]
    BS = tree.nodes.new('ShaderNodeBrightContrast')
    BS.inputs['Bright'].default_value = color.B
    BS.inputs['Contrast'].default_value = color.C
    BS.location.x -= xloc[1]
    BS.location.y -= yloc[1]
    tree.links.new(HSV.outputs['Color'], BS.inputs['Color'])
    return BS