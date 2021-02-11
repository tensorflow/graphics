/* Copyright 2020 The TensorFlow Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <memory>

#include "absl/types/span.h"
#include "macros.h"
#include "rasterizer_with_context.h"
#include "thread_safe_resource_pool.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

static tensorflow::Status GetVariablesRank(
    ::tensorflow::shape_inference::InferenceContext* c,
    tensorflow::int32* rank) {
  std::vector<std::string> variable_names, variable_kinds;
  TF_RETURN_IF_ERROR(c->GetAttr("variable_names", &variable_names));
  TF_RETURN_IF_ERROR(c->GetAttr("variable_kinds", &variable_kinds));

  std::vector<tensorflow::shape_inference::ShapeHandle> variable_values;
  TF_RETURN_IF_ERROR(c->input("variable_values", &variable_values));

  if (variable_names.size() != variable_values.size() ||
      variable_names.size() != variable_kinds.size()) {
    return tensorflow::errors::InvalidArgument(
        "The variable names, kinds, and values must have the same size.");
  }

  for (int index = 0; index < variable_kinds.size(); index++) {
    absl::string_view kind = variable_kinds[index];
    const tensorflow::shape_inference::ShapeHandle& h = variable_values[index];

    auto batch_rank = c->Rank(h);
    if (kind == "mat") {
      if (batch_rank < 2)
        return tensorflow::errors::InvalidArgument(
            "Matrix with name='", variable_names[index],
            "' has an invalid rank of ", batch_rank);
      batch_rank -= 2;
    } else if (kind == "buffer") {
      if (batch_rank < 1)
        return tensorflow::errors::InvalidArgument(
            "Buffer with name='", variable_names[index],
            "' has an invalid rank of ", batch_rank);
      batch_rank -= 1;
    }

    if (index == 0)
      *rank = batch_rank;
    else if (*rank != batch_rank)
      return tensorflow::errors::InvalidArgument(
          "Variable with name='", variable_names[index],
          "' has an invalid batch rank of ", batch_rank, "; expected ", *rank);
  }
  return tensorflow::Status::OK();
}

REGISTER_OP("Rasterize")
    .Attr("output_resolution: shape")
    .Attr("red_clear: float = 0.0")
    .Attr("green_clear: float = 0.0")
    .Attr("blue_clear: float = 0.0")
    .Attr("alpha_clear: float = 1.0")
    .Attr("depth_clear: float = 1.0")
    .Attr("enable_cull_face: bool = false")
    .Attr("vertex_shader: string")
    .Attr("fragment_shader: string")
    .Attr("geometry_shader: string")
    .Attr("variable_names: list(string)")
    .Attr("variable_kinds: list({'mat', 'buffer'})")
    .Attr("T: list({float})")
    .Input("num_points: int32")
    .Input("variable_values: T")
    .Output("rendered_image: float")
    .Doc(R"doc(
Rasterization OP that runs the program specified by the supplied vertex,
geometry and fragment shaders. Uniform variables and buffers can be passed to
the program using variable_names, variable_kinds, and variable_values.

Note that in the following, A1 to An are optional batch dimensions.

output_resolution: a 2D shape containing the width and height of the resulting
  image.
red_clear: the red component for glClear.
green_clear: the green component for glClear.
blue_clear: the blue component for glClear.
alpha_clear: the alpha component for glClear.
depth_clear: the depth value for glClearDepthf.
enable_cull_face: enable face culling.
vertex_shader: A string containing a valid vertex shader.
fragment_shader: A string containing a valid fragment shader.
geometry_shader: A string containing a valid geometry shader.
variable_names: A list of strings describing the name of each variable passed
  to the shaders. These names must map to the name of uniforms or buffers in
  the supplied shaders.
variable_kinds: A list of strings containing the type of each variable.
  Possible values for each element are `mat` and `buffer`.
num_points: The number of points to be rendered. When rasterizing a mesh, this
  number should be set to the number of vertices in the mesh.
variable_values: A list containing matrices of shape `[A1, ..., An, W, H]`
  and/or buffers of shape `[A1, ..., An, S]`, with `W` and `H` in `[1,4]` and S of
  arbitrary value. Using their associated name and kind, these values are
  mapped to the corresponding uniform or buffer in the program. Note that all
  variables must have the same batch dimensions `[A1, ..., An]`, and that
  matrices are expected to be in row-major format.
rendered_image: A tensor of shape `[A1, ..., An, width, height, 4]`, with the
  width and height defined by `output_resolution`.
    )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      tensorflow::int32 variables_rank;
      TF_RETURN_IF_ERROR(GetVariablesRank(c, &variables_rank));
      auto batch_shape = c->UnknownShapeOfRank(variables_rank);

      tensorflow::TensorShape resolution;
      TF_RETURN_IF_ERROR(c->GetAttr("output_resolution", &resolution));
      auto image_shape =
          c->MakeShape({resolution.dim_size(1), resolution.dim_size(0), 4});

      tensorflow::shape_inference::ShapeHandle output_shape;
      TF_RETURN_IF_ERROR(
          c->Concatenate(batch_shape, image_shape, &output_shape));
      c->set_output(0, output_shape);

      return tensorflow::Status::OK();
    });

class RasterizeOp : public tensorflow::OpKernel {
 public:
  explicit RasterizeOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    std::string fragment_shader;
    std::string geometry_shader;
    std::string vertex_shader;
    float red_clear = 0.0;
    float green_clear = 0.0;
    float blue_clear = 0.0;
    float alpha_clear = 1.0;
    float depth_clear = 1.0;
    bool enable_cull_face = false;

    OP_REQUIRES_OK(context, context->GetAttr("red_clear", &red_clear));
    OP_REQUIRES_OK(context, context->GetAttr("green_clear", &green_clear));
    OP_REQUIRES_OK(context, context->GetAttr("blue_clear", &blue_clear));
    OP_REQUIRES_OK(context, context->GetAttr("alpha_clear", &alpha_clear));
    OP_REQUIRES_OK(context, context->GetAttr("depth_clear", &depth_clear));
    OP_REQUIRES_OK(context,
                   context->GetAttr("enable_cull_face", &enable_cull_face));
    OP_REQUIRES_OK(context, context->GetAttr("vertex_shader", &vertex_shader));
    OP_REQUIRES_OK(context,
                   context->GetAttr("fragment_shader", &fragment_shader));
    OP_REQUIRES_OK(context,
                   context->GetAttr("geometry_shader", &geometry_shader));
    OP_REQUIRES_OK(context,
                   context->GetAttr("variable_names", &variable_names_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("variable_kinds", &variable_kinds_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_resolution", &output_resolution_));

    auto rasterizer_creator =
        [vertex_shader, geometry_shader, fragment_shader, red_clear,
         green_clear, blue_clear, alpha_clear, depth_clear, enable_cull_face,
         this](std::unique_ptr<RasterizerWithContext>* resource)
        -> tensorflow::Status {
      return RasterizerWithContext::Create(
          output_resolution_.dim_size(0), output_resolution_.dim_size(1),
          vertex_shader, geometry_shader, fragment_shader, resource, red_clear,
          green_clear, blue_clear, alpha_clear, depth_clear, enable_cull_face);
    };
    rasterizer_pool_ =
        std::unique_ptr<ThreadSafeResourcePool<RasterizerWithContext>>(
            new ThreadSafeResourcePool<RasterizerWithContext>(
                rasterizer_creator));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    tensorflow::TensorShape batch_shape;
    OP_REQUIRES_OK(context, ValidateVariables(context, &batch_shape));

    // Allocate the output images.
    tensorflow::Tensor* output_image;
    tensorflow::TensorShape output_image_shape;

    output_image_shape.AppendShape(batch_shape);
    output_image_shape.AddDim(output_resolution_.dim_size(1));
    output_image_shape.AddDim(output_resolution_.dim_size(0));
    output_image_shape.AddDim(4);
    OP_REQUIRES_OK(context, context->allocate_output(0, output_image_shape,
                                                     &output_image));

    std::unique_ptr<RasterizerWithContext> rasterizer;
    float* image_data = output_image->flat<float>().data();
    const tensorflow::int64 image_size =
        output_resolution_.dim_size(0) * output_resolution_.dim_size(1) * 4;

    OP_REQUIRES_OK(context, rasterizer_pool_->AcquireResource(&rasterizer));
    for (int i = 0; i < batch_shape.num_elements(); ++i) {
      OP_REQUIRES_OK(context, SetVariables(context, rasterizer, i));
      OP_REQUIRES_OK(context, RenderImage(context, rasterizer, image_size,
                                          image_data + i * image_size));
    }
    OP_REQUIRES_OK(context, rasterizer_pool_->ReturnResource(rasterizer));
  }

 private:
  tensorflow::Status SetVariables(
      tensorflow::OpKernelContext* context,
      std::unique_ptr<RasterizerWithContext>& rasterizer, int outer_dim);
  tensorflow::Status RenderImage(
      tensorflow::OpKernelContext* context,
      std::unique_ptr<RasterizerWithContext>& rasterizer,
      tensorflow::int64 image_size, float* image_data);
  tensorflow::Status ValidateVariables(tensorflow::OpKernelContext* context,
                                       tensorflow::TensorShape* batch_shape);

  std::unique_ptr<ThreadSafeResourcePool<RasterizerWithContext>>
      rasterizer_pool_;
  std::vector<std::string> variable_names_;
  std::vector<std::string> variable_kinds_;
  tensorflow::TensorShape output_resolution_;
};

tensorflow::Status RasterizeOp::RenderImage(
    tensorflow::OpKernelContext* context,
    std::unique_ptr<RasterizerWithContext>& rasterizer,
    const tensorflow::int64 image_size, float* image_data) {
  int num_points = context->input(0).scalar<int>()();

  TF_RETURN_IF_ERROR(rasterizer->Render(
      num_points, absl::MakeSpan(image_data, image_data + image_size)));
  return tensorflow::Status::OK();
}

tensorflow::Status RasterizeOp::SetVariables(
    tensorflow::OpKernelContext* context,
    std::unique_ptr<RasterizerWithContext>& rasterizer, int outer_dim) {
  tensorflow::OpInputList variable_values;
  TF_RETURN_IF_ERROR(context->input_list("variable_values", &variable_values));

  for (int index = 0; index < variable_names_.size(); ++index) {
    const std::string name = variable_names_[index];
    const std::string kind = variable_kinds_[index];
    const tensorflow::Tensor& value = variable_values[index];
    const tensorflow::TensorShape value_shape = value.shape();

    if (kind == "mat") {
      const int num_rows = value_shape.dim_size(value_shape.dims() - 2);
      const int num_cols = value_shape.dim_size(value_shape.dims() - 1);
      const int num_elements = num_rows * num_cols;
      const auto value_pointer = value.flat<float>().data();

      TF_RETURN_IF_ERROR(rasterizer->SetUniformMatrix(
          name, num_cols, num_rows, true,
          absl::MakeConstSpan(value_pointer + num_elements * outer_dim,
                              value_pointer + num_elements * (outer_dim + 1))));
    } else if (kind == "buffer") {
      const tensorflow::int32 buffer_length =
          value_shape.dim_size(value_shape.dims() - 1);

      const auto value_pointer = value.flat<float>().data();
      TF_RETURN_IF_ERROR(rasterizer->SetShaderStorageBuffer(
          name, absl::MakeConstSpan(
                    value_pointer + buffer_length * outer_dim,
                    value_pointer + buffer_length * (outer_dim + 1))));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status RasterizeOp::ValidateVariables(
    tensorflow::OpKernelContext* context,
    tensorflow::TensorShape* batch_shape) {
  tensorflow::OpInputList variable_values;
  TF_RETURN_IF_ERROR(context->input_list("variable_values", &variable_values));

  if (variable_names_.size() != variable_values.size() ||
      variable_names_.size() != variable_kinds_.size()) {
    return tensorflow::errors::InvalidArgument(
        "The variable names, kinds, and values must have the same size.");
  }

  bool batch_initialized = false;
  batch_shape->Clear();

  for (int index = 0; index < variable_kinds_.size(); ++index) {
    const std::string name = variable_names_[index];
    const std::string kind = variable_kinds_[index];
    const tensorflow::Tensor& value = variable_values[index];
    tensorflow::TensorShape value_batch_shape = value.shape();

    if (kind == "mat") {
      if (value_batch_shape.dims() < 2)
        return tensorflow::errors::InvalidArgument(
            "Matrix with name='", name,
            "' has an invalid shape=", value_batch_shape.DebugString());
      value_batch_shape.RemoveLastDims(2);
    } else if (kind == "buffer") {
      if (value_batch_shape.dims() < 1)
        return tensorflow::errors::InvalidArgument(
            "Buffer with name='", name,
            "' has an invalid shape=", value_batch_shape.DebugString());
      value_batch_shape.RemoveLastDims(1);
    }
    if (batch_initialized == false) {
      *batch_shape = value_batch_shape;
      batch_initialized = true;
    } else if (*batch_shape != value_batch_shape) {
      return tensorflow::errors::InvalidArgument(
          "Variable with name='", name,
          "' has an invalid batch shape=", value_batch_shape, " expected ",
          *batch_shape);
    }
  }
  return tensorflow::Status::OK();
}

// Register kernel with TF
REGISTER_KERNEL_BUILDER(Name("Rasterize").Device(tensorflow::DEVICE_CPU),
                        RasterizeOp);
