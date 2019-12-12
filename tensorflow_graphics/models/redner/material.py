import pyredner_tensorflow as pyredner
import tensorflow as tf
from typing import Union, Optional

class Material:
    """
        redner currently employs a two-layer diffuse-specular material model.
        More specifically, it is a linear blend between a Lambertian model and
        a microfacet model with Phong distribution, with Schilick's Fresnel approximation.
        It takes either constant color or 2D textures for the reflectances
        and roughness, and an optional normal map texture.
        It can also use vertex color stored in the Shape. In this case
        the model fallback to a diffuse model.

        Args
        ====
        diffuse_reflectance: Optional[Union[tf.Tensor, pyredner.Texture]]
            a float32 tensor with size 3 or [height, width, 3] or a Texture
            optional if use_vertex_color is True
        specular_reflectance: Optional[Union[tf.Tensor, pyredner.Texture]]
            a float32 tensor with size 3 or [height, width, 3] or a Texture
        roughness: Optional[Union[tf.Tensor, pyredner.Texture]]
            a float32 tensor with size 1 or [height, width, 1] or a Texture
        generic_texture: Optional[Union[tf.Tensor, pyredner.Texture]]
            a float32 tensor with dimension 1 or 3, arbitrary number of channels
            use render_g_buffer to visualize this texture
        normal_map: Optional[Union[tf.Tensor, pyredner.Texture]]
            a float32 tensor with size 3 or [height, width, 3] or a Texture
        two_sided: bool
            By default, the material only reflect lights on the side the
            normal is pointing to.
            Set this to True to make the material reflects from both sides.
        use_vertex_color: bool
            ignores the reflectances and use the vertex color as diffuse color
    """
    def __init__(self,
                 diffuse_reflectance: Optional[Union[tf.Tensor, pyredner.Texture]] = None,
                 specular_reflectance: Optional[Union[tf.Tensor, pyredner.Texture]] = None,
                 roughness: Optional[Union[tf.Tensor, pyredner.Texture]] = None,
                 generic_texture: Optional[Union[tf.Tensor, pyredner.Texture]] = None,
                 normal_map: Optional[Union[tf.Tensor, pyredner.Texture]] = None,
                 two_sided: bool = False,
                 use_vertex_color: bool = False):
        if diffuse_reflectance is None:
            diffuse_reflectance = pyredner.Texture(tf.zeros([3], dtype=tf.float32))
        if specular_reflectance is None:
            specular_reflectance = pyredner.Texture(tf.zeros([3], dtype=tf.float32))
            compute_specular_lighting = False
        else:
            compute_specular_lighting = True
        if roughness is None:
            roughness = pyredner.Texture(tf.ones([1], dtype=tf.float32))

        # Convert to constant texture if necessary
        if tf.is_tensor(diffuse_reflectance):
            diffuse_reflectance = pyredner.Texture(diffuse_reflectance)
        if tf.is_tensor(specular_reflectance):
            specular_reflectance = pyredner.Texture(specular_reflectance)
        if tf.is_tensor(roughness):
            roughness = pyredner.Texture(roughness)
        if generic_texture is not None and tf.is_tensor(generic_texture):
            generic_texture = pyredner.Texture(generic_texture)
        if normal_map is not None and tf.is_tensor(normal_map):
            normal_map = pyredner.Texture(normal_map)

        self.diffuse_reflectance = diffuse_reflectance
        self._specular_reflectance = specular_reflectance
        self.compute_specular_lighting = compute_specular_lighting
        self.roughness = roughness
        self.generic_texture = generic_texture
        self.normal_map = normal_map
        self.two_sided = two_sided
        self.use_vertex_color = use_vertex_color

    @property
    def specular_reflectance(self):
        return self._specular_reflectance

    @specular_reflectance.setter
    def specular_reflectance(self, value):
        self._specular_reflectance = value
        if value is not None:
            self.compute_specular_lighting = True
        else:
            self._specular_reflectance = pyredner.Texture(\
                tf.zeros([3], dtype=tf.float32))
            self.compute_specular_lighting = False

    def state_dict(self):
        return {
            'diffuse_reflectance': self.diffuse_reflectance.state_dict(),
            'specular_reflectance': self.specular_reflectance.state_dict(),
            'roughness': self.roughness.state_dict(),
            'generic_texture': self.generic_texture.state_dict(),
            'normal_map': self.normal_map.state_dict(),
            'two_sided': self.two_sided,
            'use_vertex_color': self.use_vertex_color
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        normal_map = state_dict['normal_map']
        out = cls(
            pyredner.Texture.load_state_dict(state_dict['diffuse_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['specular_reflectance']),
            pyredner.Texture.load_state_dict(state_dict['roughness']),
            pyredner.Texture.load_state_dict(generic_texture) if generic_texture is not None else None,
            pyredner.Texture.load_state_dict(normal_map) if normal_map is not None else None,
            state_dict['two_sided'],
            state_dict['use_vertex_color'])
        return out
