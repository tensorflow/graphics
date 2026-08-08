# Differentiable MPM for TensorFlow / PyTorch
## Installing the CUDA solver

- Install `taichi` by executing:
  ```
  wget https://raw.githubusercontent.com/yuanming-hu/taichi/master/install.py
  python3 install.py
  ```
- Make sure you are using `gcc-6`. If not, `export CXX=g++-6 CC=gcc-6`.
- Put this repo in `taichi/projects/`
- ```ti build```
- Email Yuanming when you run into any problems!


## Discretization Cheatsheet
(Assuming quadratic B-spline)
<img src="/data/images/comparison.jpg" with="1000">
