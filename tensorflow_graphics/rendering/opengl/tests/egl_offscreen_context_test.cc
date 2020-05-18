/* Copyright 2020 Google LLC

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
#include "tensorflow_graphics/rendering/opengl/egl_offscreen_context.h"

#include <GLES3/gl32.h>
#include "gtest/gtest.h"

#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

static constexpr std::array<int, 13> kDefaultConfigurationAttributes = {
    EGL_SURFACE_TYPE,
    EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE,
    EGL_OPENGL_ES2_BIT,
    EGL_BLUE_SIZE,
    8,
    EGL_GREEN_SIZE,
    8,
    EGL_RED_SIZE,
    8,
    EGL_DEPTH_SIZE,
    24,
    EGL_NONE  // The array must be terminated with that value.
};

static constexpr std::array<int, 3> kDefaultContextAttributes = {
    EGL_CONTEXT_CLIENT_VERSION,
    2,
    EGL_NONE,
};

TEST(EglOffscreenContextTest, TestCreate) {
  std::unique_ptr<EGLOffscreenContext> context;

  TF_CHECK_OK(EGLOffscreenContext::Create(
      800, 600, EGL_OPENGL_API, kDefaultConfigurationAttributes.data(),
      kDefaultContextAttributes.data(), &context));
}

TEST(EglOffscreenContextTest, TestMakeCurrentWorks) {
  std::unique_ptr<EGLOffscreenContext> context1;
  std::unique_ptr<EGLOffscreenContext> context2;

  TF_ASSERT_OK(EGLOffscreenContext::Create(
      800, 600, EGL_OPENGL_API, kDefaultConfigurationAttributes.data(),
      kDefaultContextAttributes.data(), &context1));
  TF_ASSERT_OK(EGLOffscreenContext::Create(
      400, 100, EGL_OPENGL_API, kDefaultConfigurationAttributes.data(),
      kDefaultContextAttributes.data(), &context2));
  TF_EXPECT_OK(context1->MakeCurrent());
  TF_EXPECT_OK(context2->MakeCurrent());
}

TEST(EglOffscreenContextTest, TestRelease) {
  std::unique_ptr<EGLOffscreenContext> context;

  TF_ASSERT_OK(EGLOffscreenContext::Create(
      800, 600, EGL_OPENGL_API, kDefaultConfigurationAttributes.data(),
      kDefaultContextAttributes.data(), &context));
  TF_ASSERT_OK(context->MakeCurrent());
  TF_EXPECT_OK(context->Release());
}

TEST(EglOffscreenContextTest, TestRenderClear) {
  std::unique_ptr<EGLOffscreenContext> context;
  const float kRed = 0.1;
  const float kGreen = 0.2;
  const float kBlue = 0.3;
  const float kAlpha = 1.0;
  const int kWidth = 10;
  const int kHeight = 5;
  std::vector<unsigned char> pixels(kWidth * kHeight * 4);

  TF_ASSERT_OK(EGLOffscreenContext::Create(
      kWidth, kHeight, EGL_OPENGL_API, kDefaultConfigurationAttributes.data(),
      kDefaultContextAttributes.data(), &context));
  TF_ASSERT_OK(context->MakeCurrent());
  glClearColor(kRed, kGreen, kBlue, kAlpha);
  glClear(GL_COLOR_BUFFER_BIT);
  ASSERT_EQ(glGetError(), GL_NO_ERROR);
  glReadPixels(0, 0, kWidth, kHeight, GL_RGBA, GL_UNSIGNED_BYTE, &pixels[0]);
  ASSERT_EQ(glGetError(), GL_NO_ERROR);

  for (int index = 0; index < kWidth * kHeight; ++index) {
    ASSERT_EQ(pixels[index * 4], (unsigned char)(kRed * 255.0));
    ASSERT_EQ(pixels[index * 4 + 1], (unsigned char)(kGreen * 255.0));
    ASSERT_EQ(pixels[index * 4 + 2], (unsigned char)(kBlue * 255.0));
    ASSERT_EQ(pixels[index * 4 + 3], (unsigned char)(kAlpha * 255.0));
  }
  TF_EXPECT_OK(context->Release());
}

}  // namespace
