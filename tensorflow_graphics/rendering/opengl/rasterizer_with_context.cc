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
#include "rasterizer_with_context.h"

RasterizerWithContext::RasterizerWithContext(
    std::unique_ptr<EGLOffscreenContext>&& egl_context,
    std::unique_ptr<gl_utils::Program>&& program,
    std::unique_ptr<gl_utils::RenderTargets>&& render_targets, float clear_red,
    float clear_green, float clear_blue, float clear_alpha, float clear_depth,
    bool enable_cull_face)
    : Rasterizer(std::move(program), std::move(render_targets), clear_red,
                 clear_green, clear_blue, clear_alpha, clear_depth,
                 enable_cull_face),
      egl_context_(std::move(egl_context)) {}

RasterizerWithContext::~RasterizerWithContext() {
  // Destroy the rasterizer in the correct EGL context.
  auto status = egl_context_->MakeCurrent();
  if (status != tensorflow::Status::OK())
    std::cerr
        << "~RasterizerWithContext: failure to set the context as current."
        << std::endl;
  // Reset all the members of the Rasterizer parent class before destroying the
  // context.
  this->Reset();
  // egl_context_ is destroyed here, which calls
  // egl_offscreen_context::Release().
}

tensorflow::Status RasterizerWithContext::Create(
    int width, int height, const std::string& vertex_shader_source,
    const std::string& geometry_shader_source,
    const std::string& fragment_shader_source,
    std::unique_ptr<RasterizerWithContext>* rasterizer_with_context,
    float clear_red, float clear_green, float clear_blue, float clear_alpha,
    float clear_depth, bool enable_cull_face) {
  std::unique_ptr<gl_utils::Program> program;
  std::unique_ptr<gl_utils::RenderTargets> render_targets;
  std::vector<std::pair<std::string, GLenum>> shaders;
  std::unique_ptr<EGLOffscreenContext> offscreen_context;

  TF_RETURN_IF_ERROR(EGLOffscreenContext::Create(&offscreen_context));
  TF_RETURN_IF_ERROR(offscreen_context->MakeCurrent());
  // No need to have a MakeCleanup here as EGLOffscreenContext::Release()
  // would be called on destruction of the offscreen_context object, which
  // would happen here if the whole creation process was not successful.

  shaders.push_back(std::make_pair(vertex_shader_source, GL_VERTEX_SHADER));
  shaders.push_back(std::make_pair(geometry_shader_source, GL_GEOMETRY_SHADER));
  shaders.push_back(std::make_pair(fragment_shader_source, GL_FRAGMENT_SHADER));
  TF_RETURN_IF_ERROR(gl_utils::Program::Create(shaders, &program));
  TF_RETURN_IF_ERROR(
      gl_utils::RenderTargets::Create<float>(width, height, &render_targets));
  TF_RETURN_IF_ERROR(offscreen_context->Release());
  *rasterizer_with_context =
      std::unique_ptr<RasterizerWithContext>(new RasterizerWithContext(
          std::move(offscreen_context), std::move(program),
          std::move(render_targets), clear_red, clear_green, clear_blue,
          clear_alpha, clear_depth, enable_cull_face));
  return tensorflow::Status::OK();
}

tensorflow::Status RasterizerWithContext::Render(int num_points,
                                                 absl::Span<float> result) {
  TF_RETURN_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { return this->egl_context_->Release(); });
  TF_RETURN_IF_ERROR(Rasterizer::Render(num_points, result));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return tensorflow::Status::OK();
}

tensorflow::Status RasterizerWithContext::Render(
    int num_points, absl::Span<unsigned char> result) {
  TF_RETURN_IF_ERROR(egl_context_->MakeCurrent());
  auto context_cleanup =
      MakeCleanup([this]() { return this->egl_context_->Release(); });
  TF_RETURN_IF_ERROR(Rasterizer::Render(num_points, result));
  // context_cleanup calls EGLOffscreenContext::Release here.
  return tensorflow::Status::OK();
}
