import pyredner_tensorflow as pyredner
import random
import redner
import tensorflow as tf
import math
from typing import Union, Tuple, Optional, List

def render_albedo(scene: pyredner.Scene,
                  alpha: bool = False,
                  num_samples: Union[int, Tuple[int, int]] = (16, 4),
                  seed: Optional[int] = None):
    """
        Render the diffuse albedo color of the scene.

        Args
        ====
        scene: pyredner.Scene
            pyredner Scene containing camera, geometry and material.
        alpha: bool
            If set to False, generates a 3-channel image,
            otherwise generates a 4-channel image where the
            fourth channel is alpha.
        num_samples: Union[int, Tuple[int, int]]:
            number of samples for forward and backward passes, respectively
            if a single integer is provided, use the same number of samples
            for both
        seed: Optional[int]:
            Random seed used for sampling. Randomly assigned if set to None.

        Returns
        =======
        tf.Tensor
            if alpha == True, a tensor with size [H, W, 4],
            else, a tensor with size [H, W, 3]
    """
    if seed==None:
        seed = random.randint(0, 16777216)
    channels = [redner.channels.diffuse_reflectance]
    if alpha:
        channels.append(redner.channels.alpha)
    scene_args = pyredner.serialize_scene(\
        scene = scene,
        num_samples = num_samples,
        max_bounces = 0,
        sampler_type = redner.SamplerType.sobol,
        channels = channels,
        use_secondary_edge_sampling = False)
    return pyredner.render(seed, *scene_args)

class DeferredLight:
    pass

class AmbientLight(DeferredLight):
    """
        Ambient light for deferred rendering.
    """
    def __init__(self,
                 intensity: tf.Tensor):
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        return self.intensity * albedo

class PointLight(DeferredLight):
    """
        Point light with squared distance falloff for deferred rendering.
    """
    def __init__(self,
                 position: tf.Tensor,
                 intensity: tf.Tensor):
        self.position = position
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        light_dir = self.position - position
        # the d^2 term:
        light_dist_sq = tf.reduce_sum(light_dir * light_dir, axis = 2, keepdims = True)
        light_dist = tf.sqrt(light_dist_sq)
        # Normalize light direction
        light_dir = light_dir / light_dist
        dot_l_n = tf.reduce_sum(light_dir * normal, axis = 2, keepdims = True)
        dot_l_n = tf.maximum(dot_l_n, tf.zeros_like(dot_l_n))
        return self.intensity * dot_l_n * (albedo / math.pi) / light_dist_sq 

class DirectionalLight(DeferredLight):
    """
        Directional light for deferred rendering.
    """
    def __init__(self,
                 direction: tf.Tensor,
                 intensity: tf.Tensor):
        self.direction = direction
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        # Normalize light direction
        light_dir = -self.direction / tf.norm(self.direction)
        light_dir = tf.reshape(light_dir, (1, 1, 3))
        dot_l_n = tf.reduce_sum(light_dir * normal, axis = 2, keepdims = True)
        dot_l_n = tf.maximum(dot_l_n, tf.zeros_like(dot_l_n))
        return self.intensity * dot_l_n * (albedo / math.pi)

class SpotLight(DeferredLight):
    """
        Spot light with cosine falloff for deferred rendering.
        Note that we do not provide the cosine cutoff here since it is not
        differentiable.
    """
    def __init__(self,
                 position: tf.Tensor,
                 spot_direction: tf.Tensor,
                 spot_exponent: tf.Tensor,
                 intensity: tf.Tensor):
        self.position = position
        self.spot_direction = spot_direction
        self.spot_exponent = spot_exponent
        self.intensity = intensity

    def render(self,
               position: tf.Tensor,
               normal: tf.Tensor,
               albedo: tf.Tensor):
        light_dir = self.position - position
        # Normalize light direction
        light_dir = light_dir / tf.norm(light_dir, axis = 2, keepdims = True)
        # Normalize spot direction
        spot_direction = -self.spot_direction / tf.norm(self.spot_direction)
        spot_cosine = tf.reduce_sum(light_dir * spot_direction, axis = 2, keepdims = True)
        spot_cosine = tf.maximum(spot_cosine, tf.zeros_like(spot_cosine))
        spot_factor = tf.pow(spot_cosine, self.spot_exponent)
        dot_l_n = tf.reduce_sum(light_dir * normal, axis = 2, keepdims = True)
        dot_l_n = tf.maximum(dot_l_n, tf.zeros_like(dot_l_n))
        return self.intensity * spot_factor * dot_l_n * (albedo / math.pi)

def render_deferred(scene: pyredner.Scene,
                    lights: List[DeferredLight],
                    alpha: bool = False,
                    aa_samples: int = 2,
                    seed: Optional[int] = None):
    """
        Render the scene using deferred rendering.
        (https://en.wikipedia.org/wiki/Deferred_shading)
        We generate a G-buffer image containing world-space position,
        normal, and albedo using redner, then shade the G-buffer
        using TensorFlow code. Assuming Lambertian shading and does not
        compute shadow.

        Args
        ====
        scene: pyredner.Scene
            pyredner Scene containing camera, geometry and material.
        lights: List[DeferredLight]

        alpha: bool
            If set to False, generates a 3-channel image,
            otherwise generates a 4-channel image where the
            fourth channel is alpha.
        aa_samples: int
            number of samples used for anti-aliasing at both x, y dimensions
            (e.g. if aa_samples=2, 4 samples are used)
        seed: Optional[int]:
            Random seed used for sampling. Randomly assigned if set to None.

        Returns
        =======
        tf.Tensor
            if alpha == True, a tensor with size [H, W, 4],
            else, a tensor with size [H, W, 3]
    """
    if seed==None:
        seed = random.randint(0, 16777216)

    org_res = scene.camera.resolution
    scene.camera.resolution = (org_res[0] * aa_samples,
                               org_res[1] * aa_samples)
    channels = [redner.channels.position,
                redner.channels.shading_normal,
                redner.channels.diffuse_reflectance]
    if alpha:
        channels.append(redner.channels.alpha)
    scene_args = pyredner.serialize_scene(\
        scene = scene,
        num_samples = (1, 1),
        max_bounces = 0,
        sampler_type = redner.SamplerType.sobol,
        channels = channels,
        use_secondary_edge_sampling = False)
    scene.camera.resolution = org_res
    g_buffer = pyredner.render(seed, *scene_args)
    pos = g_buffer[:, :, :3]
    normal = g_buffer[:, :, 3:6]
    albedo = g_buffer[:, :, 6:9]
    img = tf.zeros((g_buffer.shape[0], g_buffer.shape[1], 3))
    for light in lights:
        img = img + light.render(pos, normal, albedo)
    if aa_samples > 1:
        img = tf.expand_dims(img, 0) # HWC -> NHWC
        img = tf.image.resize(img, size = org_res, method = 'area', antialias = True)
        img = tf.squeeze(img, axis = 0) # NHWC -> HWC
    if alpha:
        img = tf.concat((img, g_buffer[:, :, 9:10]), axis = 2)
    return img

def render_g_buffer(scene: pyredner.Scene,
                    channels: List[redner.channels],
                    num_samples: Union[int, Tuple[int, int]] = (1, 1),
                    seed: Optional[int] = None):
    """
        Render a G buffer from the scene.

        Args
        ====
        scene: pyredner.Scene
            pyredner Scene containing camera, geometry, material, and lighting
        channels: List[pyredner.channels]
            | A list of the following channels\:
            | pyredner.channels.alpha
            | pyredner.channels.depth
            | pyredner.channels.position
            | pyredner.channels.geometry_normal
            | pyredner.channels.shading_normal
            | pyredner.channels.uv
            | pyredner.channels.diffuse_reflectance
            | pyredner.channels.specular_reflectance
            | pyredner.channels.roughness
            | pyredner.channels.generic_texture
            | pyredner.channels.vertex_color
            | pyredner.channels.shape_id
            | pyredner.channels.material_id
        num_samples: Union[int, Tuple[int, int]]
            Number of samples for forward and backward passes, respectively.
            If a single integer is provided, use the same number of samples
            for both.
        seed: Optional[int]
            Random seed used for sampling. Randomly assigned if set to None.

        Returns
        =======
        tf.Tensor
            a tensor with size [H, W, C]
    """
    if seed==None:
        seed = random.randint(0, 16777216)
    scene_args = pyredner.serialize_scene(\
        scene = scene,
        num_samples = num_samples,
        max_bounces = 0,
        sampler_type = redner.SamplerType.sobol,
        channels = channels,
        use_secondary_edge_sampling = False)
    return pyredner.render(seed, *scene_args)

def render_pathtracing(scene: pyredner.Scene,
                       alpha: bool = False,
                       max_bounces: int = 1,
                       sampler_type: redner.SamplerType = pyredner.sampler_type.sobol,
                       num_samples: Union[int, Tuple[int, int]] = (4, 4),
                       seed: Optional[int] = None):
    """
        Render a pyredner scene using pathtracing.

        Args
        ====
        scene: pyredner.Scene
            pyredner Scene containing camera, geometry, material, and lighting
        max_bounces: int
            Number of bounces for global illumination, 1 means direct lighting only.
        num_samples: int
            Number of samples per pixel for forward and backward passes.
            Can be an integer or a tuple of 2 integers.
        sampler_type: pyredner.sampler_type
            | Which sampling pattern to use? See 
              `Chapter 7 of the PBRT book <http://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction.html>`_
              for an explanation of the difference between different samplers.
            | Following samplers are supported\:
            | pyredner.sampler_type.independent
            | pyredner.sampler_type.sobol

        Returns
        =======
        torch.Tensor
            if alpha == True, a tensor with size [H, W, 4],
            else, a tensor with size [H, W, 3]
    """
    if seed==None:
        seed = random.randint(0, 16777216)
    channels = [redner.channels.radiance]
    if alpha:
        channels.append(redner.channels.alpha)
    scene_args = pyredner.serialize_scene(\
        scene = scene,
        num_samples = num_samples,
        max_bounces = max_bounces,
        sampler_type = sampler_type,
        channels = channels)
    return pyredner.render(seed, *scene_args)
