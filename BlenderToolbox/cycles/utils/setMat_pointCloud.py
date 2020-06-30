import bpy

def setMat_pointCloud(mesh, \
                ptColor, \
                ptSize): 
    # initialize a primitive sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius = 1.0, location = (1e7,1e7,1e7))
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new(name="sphereMat")
    sphere.data.materials.append(mat)
    sphere.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree
    HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
    HSVNode.inputs['Color'].default_value = ptColor.RGBA
    HSVNode.inputs['Saturation'].default_value = ptColor.S
    HSVNode.inputs['Value'].default_value = ptColor.V
    HSVNode.inputs['Hue'].default_value = ptColor.H

    # set color brightness/contrast
    BCNode = tree.nodes.new('ShaderNodeBrightContrast')
    BCNode.inputs['Bright'].default_value = ptColor.B
    BCNode.inputs['Contrast'].default_value = ptColor.C
    tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
    tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

    # init particle system
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.particle_system_add()
    bpy.data.particles["ParticleSettings"].count = len(mesh.data.vertices)
    bpy.data.particles["ParticleSettings"].frame_start = 0
    bpy.data.particles["ParticleSettings"].frame_end = 0
    bpy.data.particles["ParticleSettings"].render_type = 'OBJECT'
    bpy.data.particles["ParticleSettings"].instance_object = sphere
    bpy.data.particles["ParticleSettings"].emit_from = 'VERT'
    bpy.data.particles["ParticleSettings"].particle_size = ptSize
    bpy.data.particles["ParticleSettings"].physics_type = 'NO'
    bpy.data.particles["ParticleSettings"].use_emit_random = False
