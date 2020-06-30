import bpy
import os
# pwd = os.getcwd()
pwd = os.path.dirname(os.path.realpath(__file__))

def loadShader(shaderName, mesh):
    # switch to different shader names
    if shaderName is "EeveeToon":
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.alpha_mode = 'TRANSPARENT'
        matName = "ToonShade_EV"
        blenderFileName = 'EeveeToon.blend'
    elif shaderName is "ColoredSteel":
        matName = "Blued_Steel"
        blenderFileName = 'ColoredSteel.blend'
    elif shaderName is "Wax":
        matName = "Wax_PBR_SSS"
        blenderFileName = 'Wax.blend'
    elif shaderName is "Wood":
        matName = "UCP wood-v-1-1"
        blenderFileName = 'UCPWood.blend' # createy by Elbriga

    # load shaders to the mesh
    path = pwd + '/../../shaders/' + blenderFileName + "\\Material\\"
    bpy.ops.wm.append(filename=matName, directory=path)
    mat = bpy.data.materials.get(matName)
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    tree = mat.node_tree
    matNode = tree.nodes[-1]
    return matNode