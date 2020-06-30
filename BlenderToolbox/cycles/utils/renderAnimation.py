import bpy

def renderAnimation(outputFolder, camera, duration):
    bpy.data.scenes['Scene'].render.filepath = outputFolder
    bpy.data.scenes['Scene'].camera = camera
    bpy.data.scenes['Scene'].frame_end = duration
    bpy.ops.render.render(animation = True)
