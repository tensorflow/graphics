import redner
import tensorflow as tf

class SamplerType:
    def __init__(self):
        self.independent = redner.SamplerType.independent
        self.sobol = redner.SamplerType.sobol

sampler_type = SamplerType()
