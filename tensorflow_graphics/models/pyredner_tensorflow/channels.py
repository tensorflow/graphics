import redner
import tensorflow as tf

class Channel:
    def __init__(self):
        self.radiance = redner.channels.radiance
        self.alpha = redner.channels.alpha
        self.depth = redner.channels.depth
        self.position = redner.channels.position
        self.geometry_normal = redner.channels.geometry_normal
        self.shading_normal = redner.channels.shading_normal
        self.uv = redner.channels.uv
        self.diffuse_reflectance = redner.channels.diffuse_reflectance
        self.specular_reflectance = redner.channels.specular_reflectance
        self.roughness = redner.channels.roughness
        self.generic_texture = redner.channels.generic_texture
        self.vertex_color = redner.channels.vertex_color
        self.shape_id = redner.channels.shape_id
        self.material_id = redner.channels.material_id

channels = Channel()
