py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
      "//tensorflow_graphics/datasets/modelnet40",
      "//tensorflow_graphics/nn/layer:pointnet",
      "//tensorflow_graphics/geometry/transformation",
    ]
)

py_test(
    name = "train_test",
    srcs = ["train_test.py"],
    main = "train_test.py",
    deps = [":train"],
)