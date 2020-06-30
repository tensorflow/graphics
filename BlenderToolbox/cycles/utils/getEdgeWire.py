import bpy

def getEdgeWire(mesh, radius):

    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    mesh.select_set(True) 
    bpy.ops.mesh.delete(type='ONLY_FACE')
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh.select_set(True) 
    bpy.ops.object.convert(target='CURVE')
    mesh.data.bevel_depth = radius
    
    return mesh