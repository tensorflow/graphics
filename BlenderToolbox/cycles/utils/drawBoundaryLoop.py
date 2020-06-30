import bpy
import bmesh
from include import *

def drawBoundaryLoop(mesh, r, bdColor):
    nV = len(mesh.data.vertices)
    V = np.zeros((nV, 3), dtype = float)
    for ii in range(nV):
        V[ii,:] = mesh.matrix_local.to_3x3() @ mesh.data.vertices[int(ii)].co
        V[ii,0] += mesh.matrix_local[0][3]
        V[ii,1] += mesh.matrix_local[1][3]
        V[ii,2] += mesh.matrix_local[2][3]

    nF = len(mesh.data.polygons)
    F = np.zeros((nF, 3), dtype = int)
    for ii in range(nF):
        F[ii,:] = mesh.data.polygons[ii].vertices

    halfE = np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]), axis = 0)
    halfE = np.sort(halfE)
    E, counts = np.unique(halfE, return_counts=True, axis = 0)
    bEIdx = np.where(counts == 1)[0]
    bE = E[bEIdx,:]

    # Create bmesh 
    bdMesh = bpy.data.meshes.new('boundary') 
    bdObj = bpy.data.objects.new('objBoundary', bdMesh) 
    bpy.context.scene.collection.objects.link(bdObj)
    bm = bmesh.new()  
    bm.from_mesh(bdMesh) 

    unibE, idx = np.unique(bE,  return_inverse=True)
    bE_new = np.reshape(idx, (int(idx.shape[0]/2), 2))

    # add vertices
    VList =  []
    for ii  in range(len(unibE)):
        v = bm.verts.new( V[unibE[ii],:] )
        VList.append(v)
    
    # addedges
    for ii in range(bE_new.shape[0]):
        v1 = VList[bE_new[ii,0]]
        v2 = VList[bE_new[ii,1]]
        bm.edges.new((v1, v2))
    
    # update bmesh
    bm.to_mesh(bdMesh)

    # bevel with a circle
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = bdObj
    bdObj.select_set(state=True)
    bpy.ops.object.convert(target='CURVE')
    bpy.ops.curve.primitive_bezier_circle_add(radius=r, location=(1e5, 1e5, 1e5))
    circ = bpy.context.object
    bdObj.data.bevel_object = circ
    bpy.ops.object.shade_smooth()

    # # subdivision
    level = 2
    bpy.context.view_layer.objects.active = bdObj
    bpy.ops.object.modifier_add(type='SUBSURF')
    bdObj.modifiers["Subdivision"].render_levels = level
    bdObj.modifiers["Subdivision"].levels = level 

    # add material
    mat = bpy.data.materials.new('MeshMaterial')
    bdObj.data.materials.append(mat)
    bdObj.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree

    # init color node
    BCNode = initColorNode(tree, bdColor)

    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.7
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])
