import pyredner_tensorflow as pyredner

class Scene:
    """
        A scene is a collection of camera, geometry, materials, and light.
        Currently there are two ways to construct a scene: one is through
        lists of Shape, Material, and AreaLight. The other one is through
        a list of Object. It is more recommended to use the Object construction.
        The Shape/Material/AreaLight options are here for legacy issue.

        Args
        ====
            shapes: List[pyredner.Shape] = [],
            materials: List[pyredner.Material] = [],
            area_lights: List[pyredner.AreaLight] = [],
            objects: Optional[List[pyredner.Object]] = None,
            envmap: Optional[pyredner.EnvironmentMap] = None
    """
    def __init__(self,
                 camera,
                 shapes = [],
                 materials = [],
                 area_lights = [],
                 objects = None,
                 envmap = None):
        self.camera = camera
        self.envmap = envmap
        if objects is None:
            self.shapes = shapes
            self.materials = materials
            self.area_lights = area_lights
        else:
            # Convert objects to shapes/materials/lights
            shapes = []
            materials = []
            area_lights = []
            material_dict = {}
            current_material_id = 0
            for obj in objects:
                mid = -1
                if obj.material in material_dict:
                    mid = material_dict[obj.material]
                else:
                    mid = current_material_id
                    material_dict[obj.material] = current_material_id
                    materials.append(obj.material)
                    current_material_id += 1
                if obj.light_intensity is not None:
                    current_shape_id = len(shapes)
                    area_light = pyredner.AreaLight(shape_id = current_shape_id,
                                                    intensity = obj.light_intensity,
                                                    two_sided = obj.light_two_sided)
                    area_lights.append(area_light)
                shape = pyredner.Shape(vertices = obj.vertices,
                                       indices = obj.indices,
                                       material_id = mid,
                                       uvs = obj.uvs,
                                       normals = obj.normals,
                                       uv_indices = obj.uv_indices,
                                       normal_indices = obj.normal_indices,
                                       colors = obj.colors)
                shapes.append(shape)
            self.shapes = shapes
            self.materials = materials
            self.area_lights = area_lights

    def state_dict(self):
        return {
            'camera': self.camera.state_dict(),
            'shapes': [s.state_dict() for s in self.shapes],
            'materials': [m.state_dict() for m in self.materials],
            'area_lights': [l.state_dict() for l in self.area_lights],
            'envmap': self.envmap.state_dict() if self.envmap is not None else None
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        envmap_dict = state_dict['envmap']
        return cls(
            pyredner.Camera.load_state_dict(state_dict['camera']),
            [pyredner.Shape.load_state_dict(s) for s in state_dict['shapes']],
            [pyredner.Material.load_state_dict(m) for m in state_dict['materials']],
            [pyredner.AreaLight.load_state_dict(l) for l in state_dict['area_lights']],
            pyredner.EnvironmentMap.load_state_dict(envmap_dict) if envmap_dict is not None else None)
