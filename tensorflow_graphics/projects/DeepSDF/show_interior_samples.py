# Copyright 2020 The TensorFlow Authors
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
""" NO COMMENT NOW"""


import sys
import ctypes
import deep_sdf.data

import OpenGL.GL as gl
import pypangolin as pango

if __name__ == "__main__":

  npz_filename = sys.argv[1]

  data = deep_sdf.data.read_sdf_samples_into_ram(npz_filename)

  xyz_neg = data[1][:, 0:3].numpy().astype(ctypes.c_float)

  win = pango.CreateWindowAndBind(
      "Interior Samples | " + npz_filename, 640, 480)
  gl.glEnable(gl.GL_DEPTH_TEST)

  pm = pango.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000)
  mv = pango.ModelViewLookAt(-0, 0.5, -3, 0, 0, 0, pango.AxisY)
  s_cam = pango.OpenGlRenderState(pm, mv)

  handler = pango.Handler3D(s_cam)
  d_cam = (
      pango.CreateDisplay()
      .SetBounds(
          pango.Attach(0),
          pango.Attach(1),
          pango.Attach(0),
          pango.Attach(1),
          -640.0 / 480.0,
      )
      .SetHandler(handler)
  )

  pango.CreatePanel("ui").SetBounds(
      pango.Attach(0), pango.Attach(1), pango.Attach(0), pango.Attach(0)
  )

  while not pango.ShouldQuit():

    gl.glClear(gl.GL_COLOR_BUFFER_BIT + gl.GL_DEPTH_BUFFER_BIT)
    d_cam.Activate(s_cam)

    gl.glEnableClientState(gl.GL_VERTEX_ARRAY)

    gl.glColor3ub(255, 255, 255)

    gl.glVertexPointer(
        3, gl.GL_FLOAT, 0, xyz_neg.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))
    )

    gl.glDrawArrays(gl.GL_POINTS, 0, xyz_neg.shape[0])

    gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    pango.FinishFrame()
