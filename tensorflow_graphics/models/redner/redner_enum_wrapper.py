import tensorflow as tf
import redner

class RednerCameraType:
    __cameratypes = [
        redner.CameraType.perspective,
        redner.CameraType.orthographic,
        redner.CameraType.fisheye,
        redner.CameraType.panorama,
    ]

    @staticmethod
    def asTensor(cameratype: redner.CameraType) -> tf.Tensor:
        assert isinstance(cameratype, redner.CameraType)

        for i in range(len(RednerCameraType.__cameratypes)):
            if RednerCameraType.__cameratypes[i] == cameratype:
                return tf.constant(i)


    @staticmethod
    def asCameraType(index: tf.Tensor) -> redner.CameraType:
        try:
            cameratype = RednerCameraType.__cameratypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(RednerCameraType.__cameratypes)})')
            import sys
            sys.exit()
        else:
            return cameratype


class RednerChannels:
    __channels = [
        redner.channels.radiance,
        redner.channels.alpha,
        redner.channels.depth,
        redner.channels.position,
        redner.channels.geometry_normal,
        redner.channels.shading_normal,
        redner.channels.uv,
        redner.channels.diffuse_reflectance,
        redner.channels.specular_reflectance,
        redner.channels.roughness,
        redner.channels.generic_texture,
        redner.channels.vertex_color,
        redner.channels.shape_id,
        redner.channels.material_id
    ]

    @staticmethod
    def asTensor(channel: redner.channels) -> tf.Tensor:
        assert isinstance(channel, redner.channels)

        for i in range(len(RednerChannels.__channels)):
            if RednerChannels.__channels[i] == channel:
                return tf.constant(i)

    @staticmethod
    def asChannel(index: tf.Tensor) -> redner.channels:
        try:
            channel = RednerChannels.__channels[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(RednerChannels.__channels)})')
            import sys
            sys.exit()
        else:
            return channel


class RednerSamplerType:
    __samplertypes = [
        redner.SamplerType.independent,
        redner.SamplerType.sobol
    ]

    @staticmethod
    def asTensor(samplertype: redner.SamplerType) -> tf.Tensor:
        assert isinstance(samplertype, redner.SamplerType)

        for i in range(len(RednerSamplerType.__samplertypes)):
            if RednerSamplerType.__samplertypes[i] == samplertype:
                return tf.constant(i)


    @staticmethod
    def asSamplerType(index: tf.Tensor) -> redner.SamplerType:
        try:
            samplertype = RednerSamplerType.__samplertypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(RednerSamplerType.__samplertypes)})')
            import sys
            sys.exit()
        else:
            return samplertype