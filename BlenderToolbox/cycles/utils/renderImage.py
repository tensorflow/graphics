import bpy

def renderImage(outputPath, camera):
    bpy.data.scenes['Scene'].render.filepath = outputPath
    bpy.data.scenes['Scene'].camera = camera
    bpy.ops.render.render(write_still = True)
