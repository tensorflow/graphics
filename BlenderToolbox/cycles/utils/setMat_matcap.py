import bpy

def setMat_matcap(matcapName):
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display.shading.light = 'MATCAP'
    bpy.context.scene.display.shading.studio_light = matcapName
    
