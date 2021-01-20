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
#include "egl_offscreen_context.h"

#include <EGL/egl.h>

#include "egl_util.h"
#include "macros.h"
#include "cleanup.h"
#include "tensorflow/core/lib/core/status.h"

EGLOffscreenContext::EGLOffscreenContext(EGLContext context, EGLDisplay display,
                                         EGLSurface pixel_buffer_surface)
    : context_(context),
      display_(display),
      pixel_buffer_surface_(pixel_buffer_surface) {}

EGLOffscreenContext::~EGLOffscreenContext() { TF_CHECK_OK(Destroy()); }

tensorflow::Status EGLOffscreenContext::Create(
    std::unique_ptr<EGLOffscreenContext>* egl_offscreen_context) {
  constexpr std::array<int, 13> kDefaultConfigurationAttributes = {
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
  constexpr std::array<int, 3> kDefaultContextAttributes = {
      EGL_CONTEXT_CLIENT_VERSION,
      2,
      EGL_NONE,
  };

  return Create(0, 0, EGL_OPENGL_API, kDefaultConfigurationAttributes.data(),
                kDefaultContextAttributes.data(), egl_offscreen_context);
}

tensorflow::Status EGLOffscreenContext::Create(
    const int pixel_buffer_width, const int pixel_buffer_height,
    const EGLenum rendering_api, const EGLint* configuration_attributes,
    const EGLint* context_attributes,
    std::unique_ptr<EGLOffscreenContext>* egl_offscreen_context) {
  // Create an EGL display at device index 0.
  EGLDisplay display;

  display = CreateInitializedEGLDisplay();
  if (display == EGL_NO_DISPLAY) return TFG_INTERNAL_ERROR("EGL_NO_DISPLAY");
  auto initialize_cleanup =
      MakeCleanup([display]() { TerminateInitializedEGLDisplay(display); });

  // Set the current rendering API.
  EGLBoolean success;

  success = eglBindAPI(rendering_api);
  if (success == false) return TFG_INTERNAL_ERROR("eglBindAPI");

  // Build a frame buffer configuration.
  EGLConfig frame_buffer_configuration;
  EGLint returned_num_configs;
  const EGLint kRequestedNumConfigs = 1;

  TFG_RETURN_IF_EGL_ERROR(
      success = eglChooseConfig(display, configuration_attributes,
                                &frame_buffer_configuration,
                                kRequestedNumConfigs, &returned_num_configs));
  if (!success || returned_num_configs != kRequestedNumConfigs)
    return TFG_INTERNAL_ERROR("returned_num_configs != kRequestedNumConfigs");

  // Create a pixel buffer surface.
  EGLint pixel_buffer_attributes[] = {
      EGL_WIDTH, pixel_buffer_width, EGL_HEIGHT, pixel_buffer_height, EGL_NONE,
  };
  EGLSurface pixel_buffer_surface;

  TFG_RETURN_IF_EGL_ERROR(
      pixel_buffer_surface = eglCreatePbufferSurface(
          display, frame_buffer_configuration, pixel_buffer_attributes));
  auto surface_cleanup = MakeCleanup([display, pixel_buffer_surface]() {
    eglDestroySurface(display, pixel_buffer_surface);
  });

  // Create the EGL rendering context.
  EGLContext context;

  TFG_RETURN_IF_EGL_ERROR(
      context = eglCreateContext(display, frame_buffer_configuration,
                                 EGL_NO_CONTEXT, context_attributes));
  if (context == EGL_NO_CONTEXT) return TFG_INTERNAL_ERROR("EGL_NO_CONTEXT");

  initialize_cleanup.release();
  surface_cleanup.release();
  *egl_offscreen_context = std::unique_ptr<EGLOffscreenContext>(
      new EGLOffscreenContext(context, display, pixel_buffer_surface));
  return tensorflow::Status::OK();
}

tensorflow::Status EGLOffscreenContext::Destroy() {
  TF_RETURN_IF_ERROR(Release());
  if (eglDestroyContext(display_, context_) == false) {
    return TFG_INTERNAL_ERROR("an error occured in eglDestroyContext.");
  }
  if (eglDestroySurface(display_, pixel_buffer_surface_) == false) {
    return TFG_INTERNAL_ERROR("an error occured in eglDestroySurface.");
  }
  if (TerminateInitializedEGLDisplay(display_) == false) {
    return TFG_INTERNAL_ERROR(
        "an error occured in TerminateInitializedEGLDisplay.");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status EGLOffscreenContext::MakeCurrent() const {
  TFG_RETURN_IF_EGL_ERROR(eglMakeCurrent(display_, pixel_buffer_surface_,
                                         pixel_buffer_surface_, context_));
  return tensorflow::Status::OK();
}

tensorflow::Status EGLOffscreenContext::Release() {
  if (context_ != EGL_NO_CONTEXT && context_ == eglGetCurrentContext()) {
    TFG_RETURN_IF_EGL_ERROR(eglMakeCurrent(display_, EGL_NO_SURFACE,
                                           EGL_NO_SURFACE, EGL_NO_CONTEXT));
  }
  return tensorflow::Status::OK();
}
