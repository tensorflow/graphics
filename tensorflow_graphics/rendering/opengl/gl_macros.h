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
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_MACROS_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_MACROS_H_

#include <iostream>

#include <GLES3/gl32.h>
#include "absl/strings/str_cat.h"

#define RETURN_FALSE_IF_GL_ERROR(gl_statement)                        \
  do {                                                                \
    (gl_statement);                                                   \
    auto error = glGetError();                                        \
    if (error != GL_NO_ERROR) {                                       \
      std::cerr << absl::StrCat("GL ERROR: 0x",                       \
                                absl::Hex(error, absl::kZeroPad4)) \
                << std::endl;                                         \
      return false;                                                   \
    }                                                                 \
  } while (false)

#define RETURN_FALSE_IF_EGL_ERROR(egl_statement)                      \
  do {                                                                \
    (egl_statement);                                                  \
    auto error = eglGetError();                                       \
    if (error != EGL_SUCCESS) {                                       \
      std::cerr << absl::StrCat("EGL ERROR: 0x",                      \
                                absl::Hex(error, absl::kZeroPad4)) \
                << std::endl;                                         \
      return false;                                                   \
    }                                                                 \
  } while (false)

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_RENDERING_OPENGL_GL_MACROS_H_
