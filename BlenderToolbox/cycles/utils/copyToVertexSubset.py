import bpy
from include import *

def copyToVertexSubset(mesh, templateObj, VIdx):
    bpy.ops.object.select_all(action = 'DESELECT')
    templateObj.select_set(True)
    bpy.context.view_layer.objects.active = templateObj
    for ii in VIdx:
        Vloc = mesh.matrix_world @ mesh.data.vertices[int(ii)].co
        bpy.ops.object.duplicate({"object" : templateObj}, linked=True)
        objCopy = bpy.context.object
        objCopy.location = Vloc