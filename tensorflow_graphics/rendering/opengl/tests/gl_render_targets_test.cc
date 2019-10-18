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
#include "tensorflow_graphics/rendering/opengl/gl_render_targets.h"

#include "gtest/gtest.h"
#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

template <typename T>
class RenderTargetsInterfaceTest : public ::testing::Test {};
using valid_render_target_types = ::testing::Types<float, unsigned char>;
TYPED_TEST_CASE(RenderTargetsInterfaceTest, valid_render_target_types);

TYPED_TEST(RenderTargetsInterfaceTest, TestRenderClear) {
  std::unique_ptr<EGLOffscreenContext> context;
  const float kRed = 0.1;
  const float kGreen = 0.2;
  const float kBlue = 0.3;
  const float kAlpha = 1.0;
  const int kWidth = 10;
  const int kHeight = 5;
  float max_gl_value = 255.0f;

  if (typeid(TypeParam) == typeid(float)) max_gl_value = 1.0f;

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());

  std::unique_ptr<gl_utils::RenderTargets<TypeParam>> render_targets;
  TF_ASSERT_OK(gl_utils::RenderTargets<TypeParam>::Create(kWidth, kHeight,
                                                          &render_targets));
  glClearColor(kRed, kGreen, kBlue, kAlpha);
  glClear(GL_COLOR_BUFFER_BIT);
  ASSERT_EQ(glGetError(), GL_NO_ERROR);
  std::vector<TypeParam> pixels(kWidth * kHeight * 4);
  TF_ASSERT_OK(render_targets->CopyPixelsInto(absl::MakeSpan(pixels)));

  for (int index = 0; index < kWidth * kHeight; ++index) {
    ASSERT_EQ(pixels[index * 4], TypeParam(kRed * max_gl_value));
    ASSERT_EQ(pixels[index * 4 + 1], TypeParam(kGreen * max_gl_value));
    ASSERT_EQ(pixels[index * 4 + 2], TypeParam(kBlue * max_gl_value));
    ASSERT_EQ(pixels[index * 4 + 3], TypeParam(kAlpha * max_gl_value));
  }
  TF_EXPECT_OK(context->Release());
}

TYPED_TEST(RenderTargetsInterfaceTest, TestReadFails) {
  std::unique_ptr<EGLOffscreenContext> context;
  const int kWidth = 10;
  const int kHeight = 5;

  TF_ASSERT_OK(EGLOffscreenContext::Create(&context));
  TF_ASSERT_OK(context->MakeCurrent());
  std::unique_ptr<gl_utils::RenderTargets<TypeParam>> render_targets;
  TF_ASSERT_OK(gl_utils::RenderTargets<TypeParam>::Create(kWidth, kHeight,
                                                          &render_targets));
  std::vector<TypeParam> pixels(kWidth * kHeight * 3);
  EXPECT_NE(render_targets->CopyPixelsInto(absl::MakeSpan(pixels)),
            tensorflow::Status::OK());
}

}  // namespace
