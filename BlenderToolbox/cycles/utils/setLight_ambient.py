import bpy

def setLight_ambient(color = (0,0,0,1)):
	bpy.data.scenes[0].world.use_nodes = True
	bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color