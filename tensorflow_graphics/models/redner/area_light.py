import pyredner_tensorflow as pyredner
import tensorflow as tf

class AreaLight:
    """
        A mesh-based area light that points to a shape and assigns intensity.

        Args
        ----------
        shape_id: int

        intensity: tf.Tensor
            1-d tensor with size 3 and type float32
        two_sided: bool
            is the light emitting light from the two sides of the faces?
    """

    def __init__(self,
                 shape_id: int,
                 intensity: tf.Tensor,
                 two_sided: bool = False):
        self.shape_id = shape_id
        self.intensity = intensity
        self.two_sided = two_sided

    def state_dict(self):
        return {
            'shape_id': self.shape_id,
            'intensity': self.intensity,
            'two_sided': self.two_sided
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        return cls(
            state_dict['shape_id'],
            state_dict['intensity'],
            state_dict['two_sided'])
