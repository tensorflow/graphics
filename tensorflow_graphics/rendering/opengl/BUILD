# OpenGL functionalities for tf-graphics.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

genrule(
    name = "rasterizer_op",
    srcs = [
        "cleanup.h",
        "egl_offscreen_context.cc",
        "egl_offscreen_context.h",
        "egl_util.cc",
        "egl_util.h",
        "gl_program.cc",
        "gl_program.h",
        "gl_render_targets.cc",
        "gl_render_targets.h",
        "gl_shader_storage_buffer.cc",
        "gl_shader_storage_buffer.h",
        "macros.h",
        "rasterizer.cc",
        "rasterizer.h",
        "rasterizer_op.cc",
        "rasterizer_with_context.cc",
        "rasterizer_with_context.h",
        "thread_safe_resource_pool.h",
    ],
    outs = ["rasterizer_op.so"],
    cmd = "TF_CFLAGS=$$(python -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))');\
           TF_LFLAGS=$$(python -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))');\
           g++ -std=c++14 -shared $(SRCS) -o $(OUTS) -fPIC $${TF_CFLAGS[@]} $${TF_LFLAGS[@]}\
           -DUSE_OZONE -Wl,-L/usr/lib/x86_64-linux-gnu/mesa-egl -Wl,-L/usr/lib/x86_64-linux-gnu -Wl,-lEGL -Wl,-lGLESv2 -O2;\
           VAR_OUTS=$(OUTS);\
           VAR_GENDIR=$(GENDIR);\
           cp $(OUTS) $(BASEDIR)/$${VAR_OUTS#$$VAR_GENDIR}",
)
