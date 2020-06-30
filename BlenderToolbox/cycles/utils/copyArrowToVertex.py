import bpy
import math
import numpy as np

def copyArrowToVertex(mesh, tmpArrow, VIdx, VNs = None):
    bpy.ops.object.select_all(action = 'DESELECT')
    tmpArrow.select_set(True)
    bpy.context.view_layer.objects.active = tmpArrow
    for ii in range(len(VIdx)):
        print("copy progress " + str(ii) + "/" + str(len(VIdx)))
        v = VIdx[ii]
        pos = mesh.matrix_world @ mesh.data.vertices[v].co
        bpy.ops.object.duplicate({"object" : tmpArrow}, linked=True)
        objCopy = bpy.context.object
        objCopy.location = pos

        # if not prescribed normals, then draw along vertex normal
        if VNs is None: 
            VN = mesh.matrix_world @ mesh.data.vertices[v].normal
            VN = VN.normalized()
        else:
            VN = VNs[ii,:]
            VN = VN / np.linalg.norm(VN)
        phi = math.atan2(VN[1], VN[0]) 
        theta = math.acos(VN[2]) 
        objCopy.rotation_euler[1] = theta 
        objCopy.rotation_euler[2] = phi 

    
        