# CPU rasterizer functionalities for tf-graphics.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

genrule(
    name = "rasterizer_op",
    srcs = [
        "rasterize_triangles_impl.cc",
        "rasterize_triangles_impl.h",
        "rasterize_triangles_op.cc",
    ],
    outs = ["rasterizer_op.so"],
    cmd = "TF_CFLAGS=$$(python -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))');\
           TF_LFLAGS=$$(python -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))');\
           g++ -std=c++14 -shared $(SRCS) -o $(OUTS) -fPIC $${TF_CFLAGS[@]} $${TF_LFLAGS[@]}\
           -DUSE_OZONE -Wl,-L/usr/lib/x86_64-linux-gnu -O2;\
           VAR_OUTS=$(OUTS);\
           VAR_GENDIR=$(GENDIR);\
           cp $(OUTS) $(BASEDIR)/$${VAR_OUTS#$$VAR_GENDIR}",
)
