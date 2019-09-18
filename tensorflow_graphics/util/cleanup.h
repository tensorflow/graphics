#Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#ifndef THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_UTIL_CLEANUP_H_
#define THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_UTIL_CLEANUP_H_

#include <iostream>

// Object that calls a stored cleanup functor on destruction.
template <typename F>
class Cleanup {
 public:
  template <typename G>
  Cleanup(G&& cleanup_function);
  // Calls the cleanup function if Release was not called, or does nothing
  // alternatively.
  ~Cleanup();
  // Makes the destructor do nothing instead of calling the cleanup functor.
  void Release();

 private:
  Cleanup() = delete;
  Cleanup(const Cleanup&) = delete;
  Cleanup(Cleanup&&) = delete;
  Cleanup& operator=(const Cleanup&) = delete;
  Cleanup& operator=(Cleanup&&) = delete;

  bool cleanup_needed_;
  F cleanup_function_;
};

template <typename F>
template <typename G>
Cleanup<F>::Cleanup(G&& cleanup_function)
    : cleanup_needed_(true), cleanup_function_(cleanup_function) {}

template <typename F>
Cleanup<F>::~Cleanup() {
  if (cleanup_needed_) cleanup_function_();
}

template <typename F>
void Cleanup<F>::Release() {
  cleanup_needed_ = false;
}

template <typename F>
Cleanup<F> MakeCleanup(F&& cleanup_function) {
  return Cleanup<F>(cleanup_function);
}

#endif  // THIRD_PARTY_PY_TENSORFLOW_GRAPHICS_UTIL_CLEANUP_H_
