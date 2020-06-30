import bpy

def shadowThreshold(alphaThreshold, interpolationMode = 'CARDINAL'):
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    RAMP = tree.nodes.new('CompositorNodeValToRGB')
    RAMP.color_ramp.elements[0].color[3] = 0
    RAMP.color_ramp.elements[0].position = alphaThreshold
    RAMP.color_ramp.interpolation = interpolationMode

    REND = tree.nodes["Render Layers"]
    OUT = tree.nodes["Composite"]
    tree.links.new(REND.outputs[1], RAMP.inputs[0])
    tree.links.new(RAMP.outputs[1], OUT.inputs[1])