workspace(name = "tensorflow_graphics")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:tf.bzl", "tf_configure")


http_archive(
    name = "bazel_skylib",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.0/bazel-skylib-1.0.0.tar.gz",
    sha256 = "e72747100a8b6002992cc0bf678f6279e71a3fd4a88cab3371ace6c73432be30",
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "9d359cc1b508082d8ba309ba085da6ecec85e7a4d5bd08f8db9666ee39a85529",
    strip_prefix = "rules_closure-0.9.0",
    url = "https://github.com/bazelbuild/rules_closure/archive/0.9.0.zip",
)

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-master",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/master.zip"],
)

http_archive(
    name = "com_google_absl",
    strip_prefix = "abseil-cpp-43ef2148c0936ebf7cb4be6b19927a9d9d145b8f",
    url = "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
    sha256 = "acd93f6baaedc4414ebd08b33bebca7c7a46888916101d8c0b8083573526d070",
)

# Set up dependency to tensorflow pip package.
tf_configure()

