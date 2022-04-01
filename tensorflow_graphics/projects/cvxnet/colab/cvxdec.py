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
"""A simple 2D demo of the differentiable convex function."""

# --- being forgiving as this is a colab
# pylint: skip-file

#%% Load the data (from point picker)
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# --- Equations of hyperplanes in the 'hyperplanes.png' image
h0 = np.array([(223.84848484848487, 55.04545454545456),
               (97.78787878787875, 91.4848484848485)])
h1 = np.array([(96.80303030303028, 91.4848484848485),
               (62.333333333333286, 239.21212121212125)])
h2 = np.array([(62.333333333333286, 239.21212121212125),
               (134.2272727272727, 311.1060606060606)])
h3 = np.array([(134.2272727272727, 311.1060606060606),
               (264.22727272727275, 161.40909090909093)])
h4 = np.array([(264.22727272727275, 161.40909090909093),
               (223.84848484848487, 55.04545454545456)])
h5 = np.array([(234.6818181818182, 327.8484848484849),
               (333.1666666666667, 159.43939393939394)])
hs = [h0, h1, h2, h3, h4, h5]

# --- Load base image
img = mpimg.imread('hyperplanes.png')

if False:
  #--- Check lines match PNG
  plt.figure(0)
  imgplot = plt.imshow(img)

  def ploth(h):
    plt.plot(h[0][0], h[0][1], '.r')
    plt.plot(h[1][0], h[1][1], '.r')

  for h in hs:
    ploth(h)


def pointnormal(h):
  ROT = np.array([[0, -1], [1, 0]])
  p1 = np.array(h[0][:])
  p2 = np.array(h[1][:])
  n = (p2 - p1) / np.linalg.norm(p2 - p1)
  return p1, np.dot(ROT, n)


#--- Define sampling domain
x = np.linspace(0, 364, 364)
y = np.linspace(0, 364, 364)
XX, YY = np.meshgrid(x, y)

#--- Compute the SDFs
D = np.zeros((len(hs), img.shape[0], img.shape[1]))
for i, hi in enumerate(hs):
  p0, n0 = pointnormal(hi)
  XY = np.stack([XX, YY])
  p0 = np.reshape(p0, [2, 1, 1])  # (2,1,1)
  n0 = np.reshape(n0, [2, 1, 1])
  off = (XY - p0)  #< broadcat (2,W,H)
  d = np.linalg.norm(off, axis=0)
  d = np.einsum('i...,i...', n0, off)
  D[i, ...] = d

# softmax = lambda x, delta: np.exp(delta*x) / np.sum(np.exp(delta*x), axis=0)
softmax = lambda x, delta=1: np.log(np.sum(np.exp(delta * x), axis=0)) / delta
Dmax = softmax(D)

# Dmax = D.max(axis=0)
D_clim = np.maximum(D.max(), -D.min())
Dmax_clim = np.maximum(Dmax.max(), -Dmax.min())
Dshift = Dmax  #< ?what was this?

sigmoid = lambda x, sigma: 1 / (1 + np.exp(sigma * x))
Dout = sigmoid(Dshift, 1 / 10.)

#%%
#--- individual
get_ipython().system('mkdir cvxdec')
for i, hi in enumerate(hs):
  d = D[i, ...]
  plt.figure(i)
  plt.imshow(d, cmap=plt.get_cmap('coolwarm'), clim=(-D_clim, +D_clim))
  plt.contour(d, [0])
  plt.axis('off')
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(
      'cvxdec/sdf_{}.png'.format(i), bbox_inches='tight', pad_inches=-.1)

#%%
#--- Display a single one + the colormap beside it
for i, hi in enumerate(hs):
  d = D[i, ...]
  plt.figure(i)
  imaxis = plt.imshow(d, cmap=plt.get_cmap('coolwarm'), clim=(-D_clim, +D_clim))
  plt.contour(d, [0])
  plt.gcf().colorbar(imaxis)
  plt.axis('off')
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())
  plt.savefig(
      'cvxdec/sdf_{}_cmap.png'.format(i), bbox_inches='tight', pad_inches=-.1)
  break

#%%
#--- max / union
plt.figure()
imaxis = plt.imshow(
    Dmax, cmap=plt.get_cmap('coolwarm'), clim=(-Dmax_clim, +Dmax_clim))
plt.contour(Dmax, [0])
plt.axis('off')
plt.gcf().colorbar(imaxis)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig('cvxdec/maxoperator_cmap.png', bbox_inches='tight', pad_inches=0)
plt.show()

#%%
#--- max / union with different thresholds
Dmax_news = list()
for idelta, delta in enumerate([0.040, 0.060, 0.080, 1]):
  Dmax_new = softmax(D, delta)
  Dmax_news.append(Dmax_new)
  if True:
    print(delta)
    plt.figure(idelta, frameon=False)
    imaxis = plt.imshow(
        Dmax, cmap=plt.get_cmap('coolwarm'), clim=(-Dmax_clim, +Dmax_clim))
    plt.contour(Dmax_new, [0])
    # plt.gcf().colorbar(imaxis)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(
        'cvxdec/softmax_{}.png'.format(delta),
        bbox_inches='tight',
        pad_inches=-.1)
    plt.show()

#%%
#--- sigmoid
Dshift = Dmax_news[2]
for isigma, sigma in enumerate([1 / 5]):
  Dout = sigmoid(Dshift, sigma)
  if True:
    plt.figure(isigma, frameon=False)
    imaxis = plt.imshow(Dout, cmap=plt.get_cmap('coolwarm'), clim=(0, 1))
    plt.contour(Dout, [0.5])
    plt.axis('off')
    # plt.gcf().colorbar(imaxis)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(
        'cvxdec/sigmoid_{}.png'.format(sigma),
        bbox_inches='tight',
        pad_inches=-.1)
    plt.show()

#%%
#--- 2D visualization
plt.figure()
plt.plot(Dout[182, :])
