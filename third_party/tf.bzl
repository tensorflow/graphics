"""Set up dependency to tensorflow pip package."""

def _find_tf_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            "python3",
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow. Please install TensorFlow pip package first.")
    return exec_result.stdout.splitlines()[-1]

def _find_tf_lib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            "python3",
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_lib())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow. Please install TensorFlow pip package first.")
    return exec_result.stdout.splitlines()[-1]

def _eigen_archive_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tf_includes/Eigen/**/*.h",
                 "tf_includes/Eigen/**",
                 "tf_includes/unsupported/Eigen/**/*.h",
                 "tf_includes/unsupported/Eigen/**"]),
    # https://groups.google.com/forum/#!topic/bazel-discuss/HyyuuqTxKok
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _absl_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/absl",
        "absl",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["absl/**/*.h",
                 "absl/**/*.inc"]),
    includes = ["absl"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _zlib_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/zlib",
        "zlib",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["zlib/**/*.h"]),
    includes = ["zlib"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _protobuf_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tf_includes/google/protobuf/*.h",
                 "tf_includes/google/protobuf/*.inc",
                 "tf_includes/google/protobuf/**/*.h",
                 "tf_includes/google/protobuf/**/*.inc"]),
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _tensorflow_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tensorflow_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tensorflow_includes/**/*.h",
                 "tensorflow_includes/third_party/eigen3/**"]),
    includes = ["tensorflow_includes"],
    deps = ["@absl_includes//:includes",
            "@eigen_archive//:includes",
            "@protobuf_archive//:includes",
            "@zlib_includes//:includes",],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _tensorflow_solib_repo_impl(repo_ctx):
    tf_lib_path = _find_tf_lib_path(repo_ctx)
    repo_ctx.symlink(tf_lib_path, "tensorflow_solib")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "framework_lib",
    srcs = ["tensorflow_solib/libtensorflow_framework.so.2"],
    visibility = ["//visibility:public"],
)
""",
    )

def tf_configure():
    """Autoconf pre-installed tensorflow pip package."""
    make_eigen_repo = repository_rule(implementation = _eigen_archive_repo_impl)
    make_eigen_repo(name = "eigen_archive")
    make_absl_repo = repository_rule(
        implementation = _absl_includes_repo_impl,
    )
    make_absl_repo(name = "absl_includes")
    make_zlib_repo = repository_rule(
        implementation = _zlib_includes_repo_impl,
    )
    make_zlib_repo(name = "zlib_includes")
    make_protobuf_repo = repository_rule(
        implementation = _protobuf_includes_repo_impl,
    )
    make_protobuf_repo(name = "protobuf_archive")
    make_tfinc_repo = repository_rule(
        implementation = _tensorflow_includes_repo_impl,
    )
    make_tfinc_repo = repository_rule(
        implementation = _tensorflow_includes_repo_impl,
    )
    make_tfinc_repo(name = "tensorflow_includes")

    make_tflib_repo = repository_rule(
        implementation = _tensorflow_solib_repo_impl,
    )
    make_tflib_repo(name = "tensorflow_solib")
