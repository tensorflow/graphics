/* Copyright 2019 Google LLC

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
#include "tensorflow_graphics/rendering/opengl/macros.h"
#include "tensorflow_graphics/rendering/opengl/rasterizer_with_context.h"
#include "tensorflow_graphics/rendering/opengl/thread_safe_resource_pool.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

REGISTER_OP("Rasterize")
    .Attr("output_resolution: shape")
    .Attr("red_clear: float = 0.0")
    .Attr("green_clear: float = 0.0")
    .Attr("blue_clear: float = 0.0")
    .Attr("depth_clear: float = 1.0")
    .Attr("vertex_shader: string")
    .Attr("fragment_shader: string")
    .Attr("geometry_shader: string")
    .Attr("variable_names: list(string)")
    .Attr("variable_kinds: list({'mat', 'buffer'})")
    .Attr("T: list({float})")
    .Input("num_points: int32")
    .Input("variable_values: T")
    .Output("rendered_image: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      // Error handling.
      std::vector<std::string> variable_names;
      TF_RETURN_IF_ERROR(c->GetAttr("variable_names", &variable_names));

      std::vector<tensorflow::shape_inference::ShapeHandle> variable_values;
      TF_RETURN_IF_ERROR(c->input("variable_values", &variable_values));

      if (variable_names.size() != variable_values.size())
        return tensorflow::errors::InvalidArgument(
            "The number of elements in variable_names, and variable_values.");

      // Defines the shape of the output tensor.
      tensorflow::TensorShape resolution;
      TF_RETURN_IF_ERROR(c->GetAttr("output_resolution", &resolution));
      c->set_output(
          0, c->MakeShape({resolution.dim_size(0), resolution.dim_size(1), 4}));

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
    float depth_clear = 1.0;

    OP_REQUIRES_OK(context, context->GetAttr("red_clear", &red_clear));
    OP_REQUIRES_OK(context, context->GetAttr("green_clear", &green_clear));
    OP_REQUIRES_OK(context, context->GetAttr("blue_clear", &blue_clear));
    OP_REQUIRES_OK(context, context->GetAttr("depth_clear", &depth_clear));
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
    OP_REQUIRES(context, variable_names_.size() == variable_kinds_.size(),
                tensorflow::errors::InvalidArgument(
                    "The variable names and kinds must have the same size"));

    auto rasterizer_creator =
        [vertex_shader, geometry_shader, fragment_shader, red_clear,
         green_clear, blue_clear, depth_clear,
         this](std::unique_ptr<RasterizerWithContext<float>>* resource)
        -> tensorflow::Status {
      return RasterizerWithContext<float>::Create(
          output_resolution_.dim_size(1), output_resolution_.dim_size(0),
          vertex_shader, geometry_shader, fragment_shader, resource, red_clear,
          green_clear, blue_clear, depth_clear);
    };
    rasterizer_pool_ =
        std::unique_ptr<ThreadSafeResourcePool<RasterizerWithContext<float>>>(
            new ThreadSafeResourcePool<RasterizerWithContext<float>>(
                rasterizer_creator));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    std::unique_ptr<RasterizerWithContext<float>> rasterizer;

    OP_REQUIRES_OK(context, rasterizer_pool_->AcquireResource(&rasterizer));
    OP_REQUIRES_OK(context, SetVariables(context, rasterizer));
    OP_REQUIRES_OK(context, RenderImage(context, rasterizer));
    OP_REQUIRES_OK(context, rasterizer_pool_->ReturnResource(rasterizer));
  }

 private:
  tensorflow::Status SetVariables(
      tensorflow::OpKernelContext* context,
      std::unique_ptr<RasterizerWithContext<float>>& rasterizer);
  tensorflow::Status RenderImage(
      tensorflow::OpKernelContext* context,
      std::unique_ptr<RasterizerWithContext<float>>& rasterizer);

  std::unique_ptr<ThreadSafeResourcePool<RasterizerWithContext<float>>>
      rasterizer_pool_;
  std::vector<std::string> variable_names_;
  std::vector<std::string> variable_kinds_;
  tensorflow::TensorShape output_resolution_;
};

tensorflow::Status RasterizeOp::RenderImage(
    tensorflow::OpKernelContext* context,
    std::unique_ptr<RasterizerWithContext<float>>& rasterizer) {
  tensorflow::Tensor* output_image;
  tensorflow::TensorShape output_image_shape(
      {output_resolution_.dim_size(0), output_resolution_.dim_size(1), 4});

  TF_RETURN_IF_ERROR(
      context->allocate_output(0, output_image_shape, &output_image));
  auto image_data = output_image->flat<float>();
  int num_points = context->input(0).scalar<int>()();
  TF_RETURN_IF_ERROR(rasterizer->Render(
      num_points, absl::MakeSpan(image_data.data(),
                                 image_data.data() + image_data.size())));
  return tensorflow::Status::OK();
}

tensorflow::Status RasterizeOp::SetVariables(
    tensorflow::OpKernelContext* context,
    std::unique_ptr<RasterizerWithContext<float>>& rasterizer) {
  tensorflow::OpInputList variable_values;
  TF_RETURN_IF_ERROR(context->input_list("variable_values", &variable_values));

  if (variable_names_.size() != variable_values.size()) {
    return tensorflow::errors::InvalidArgument(
        "The variable names and values must have the same size.");
  }

  for (int index = 0; index < variable_names_.size(); ++index) {
    const std::string name = variable_names_[index];
    const std::string kind = variable_kinds_[index];
    const tensorflow::Tensor& value = variable_values[index];
    const tensorflow::TensorShape value_shape = value.shape();
    const tensorflow::DataType value_dtype = value.dtype();

    if (kind == "mat" && value_dtype == tensorflow::DT_FLOAT &&
        value_shape.dims() == 2) {
      const int num_rows = value_shape.dim_size(0);
      const int num_cols = value_shape.dim_size(1);
      TF_RETURN_IF_ERROR(rasterizer->SetUniformMatrix(
          name, num_cols, num_rows, true,
          absl::MakeConstSpan(value.flat<float>())));
    } else if (kind == "buffer" && value_dtype == tensorflow::DT_FLOAT &&
               value_shape.dims() == 1) {
      TF_RETURN_IF_ERROR(rasterizer->SetShaderStorageBuffer(
          name, absl::MakeConstSpan(value.flat<float>())));
    } else {
      return tensorflow::errors::InvalidArgument(
          "Don't know how to handle variable with name='", name,
          "', kind=", kind, " shape=", value_shape.DebugString(),
          " and type=", value_dtype);
    }
  }
  return tensorflow::Status::OK();
}

// Register kernel with TF
REGISTER_KERNEL_BUILDER(Name("Rasterize").Device(tensorflow::DEVICE_CPU),
                        RasterizeOp);
