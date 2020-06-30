import bpy

def setLight_threePoints(
    radius = 4, 
    height = 10, 
    intensity = 1700, 
    softness = 6,
    keyLoc = 'left'):
    if keyLoc == 'left':
        bpy.ops.object.light_add(type='POINT', radius=softness, location=(radius,0,height))
        KeyL = bpy.data.lights['Point']
        KeyL.energy = intensity
        bpy.ops.object.light_add(type='POINT', radius=softness, location=(0,radius,0.6*height))
        FillL = bpy.data.lights['Point.001']
        FillL.energy = intensity * 0.5
        bpy.ops.object.light_add(type='POINT', radius=softness, location=(0,-radius,height))
    else:
        bpy.ops.object.light_add(type='POINT', radius=softness, location=(0,radius,height))
        KeyL = bpy.data.lights['Point']
        KeyL.energy = intensity
        bpy.ops.object.light_add(type='POINT', radius=softness, location=(radius,0,0.6*height))
        FillL = bpy.data.lights['Point.001']
        FillL.energy = intensity * 0.5
        bpy.ops.object.light_add(type='POINT', radius=softness, location=(-radius,0,height))
    RimL = bpy.data.lights['Point.002']
    RimL.energy = intensity * 0.1
    return [KeyL, FillL, RimL]