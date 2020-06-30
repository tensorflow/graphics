import bpy

def recalculateNormals(mesh):
    bpy.ops.object.select_all(action = 'DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.set_normals_from_faces()
    bpy.ops.object.editmode_toggle()