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
#include "tensorflow_graphics/rendering/opengl/gl_render_targets.h"

#include "gtest/gtest.h"
#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"
#include "tensorflow_graphics/rendering/opengl/macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

template <typename CreateT, typename CopyT>
tensorflow::Status TestRenderClear() {
  std::unique_ptr<EGLOffscreenContext> context;
  const float kRed = 0.2;
  const float kGreen = 0.4;
  const float kBlue = 0.6;
  const float kAlpha = 1.0;
  const int kWidth = 10;
  const int kHeight = 5;
  float max_gl_value = 255.0f;

  if (typeid(CopyT) == typeid(float)) max_gl_value = 1.0f;

  TF_RETURN_IF_ERROR(EGLOffscreenContext::Create(&context));
  TF_RETURN_IF_ERROR(context->MakeCurrent());

  std::unique_ptr<gl_utils::RenderTargets> render_targets;
  TF_RETURN_IF_ERROR(gl_utils::RenderTargets::Create<CreateT>(kWidth, kHeight,
                                                              &render_targets));
  glClearColor(kRed, kGreen, kBlue, kAlpha);
  TFG_RETURN_IF_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT));
  std::vector<CopyT> pixels(kWidth * kHeight * 4);
  TF_RETURN_IF_ERROR(render_targets->CopyPixelsInto(absl::MakeSpan(pixels)));

  for (int index = 0; index < kWidth * kHeight; ++index) {
    if (pixels[index * 4] != CopyT(kRed * max_gl_value))
      return tensorflow::errors::InvalidArgument(
          "pixels[index * 4] != CopyT(kRed * max_gl_value)");
    if (pixels[index * 4 + 1] != CopyT(kGreen * max_gl_value))
      return tensorflow::errors::InvalidArgument(
          "pixels[index * 4] != CopyT(kGreen * max_gl_value)");
    if (pixels[index * 4 + 2] != CopyT(kBlue * max_gl_value))
      return tensorflow::errors::InvalidArgument(
          "pixels[index * 4] != CopyT(kBlue * max_gl_value)");
    if (pixels[index * 4 + 3] != CopyT(kAlpha * max_gl_value))
      return tensorflow::errors::InvalidArgument(
          "pixels[index * 4] != CopyT(kAlpha * max_gl_value)");
  }
  TF_RETURN_IF_ERROR(context->Release());
  return tensorflow::Status::OK();
}

TEST(RenderTargetsInterfaceTest, TestRenderClear) {
  TF_EXPECT_OK((TestRenderClear<float, float>()));
  TF_EXPECT_OK((TestRenderClear<unsigned char, unsigned char>()));
  TF_EXPECT_OK((TestRenderClear<float, unsigned char>()));
  TF_EXPECT_OK((TestRenderClear<unsigned char, float>()));
}

template <typename T>
class RenderTargetsInterfaceTest : public ::testing::Test {};
using valid_render_target_types = ::testing::Types<float, unsigned char>;
TYPED_TEST_CASE(RenderTargetsInterfaceTest, valid_render_target_types);

TYPED_TEST(RenderTargetsInterfaceTest, TestReadFails) {
  std::unique_ptr<EGLOffscreenContext> context;
  const int kWidth = 10;
  const int kHeight = 5;

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());
  std::unique_ptr<gl_utils::RenderTargets> render_targets;
  TF_ASSERT_OK(gl_utils::RenderTargets::Create<TypeParam>(kWidth, kHeight,
                                                          &render_targets));
  std::vector<TypeParam> pixels(kWidth * kHeight * 3);
  EXPECT_NE(render_targets->CopyPixelsInto(absl::MakeSpan(pixels)),
            tensorflow::Status::OK());
}

}  // namespace
