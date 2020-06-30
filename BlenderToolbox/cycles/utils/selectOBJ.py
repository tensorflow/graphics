import bpy

def selectOBJ(mesh):
    bpy.ops.object.select_all(action = 'DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh