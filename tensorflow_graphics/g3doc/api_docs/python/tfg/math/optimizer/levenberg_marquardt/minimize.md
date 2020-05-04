<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfg.math.optimizer.levenberg_marquardt.minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfg.math.optimizer.levenberg_marquardt.minimize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/optimizer/levenberg_marquardt.py">View source</a>



Minimizes a set of residuals in the least-squares sense.

```python
tfg.math.optimizer.levenberg_marquardt.minimize(
    residuals, variables, max_iterations, regularizer=1e-20,
    regularizer_multiplier=10.0, callback=None, name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`residuals`</b>: A residual or a list/tuple of residuals. A residual is a Python
  `callable`.
* <b>`variables`</b>: A variable or a list or tuple of variables defining the starting
  point of the minimization.
* <b>`max_iterations`</b>: The maximum number of iterations.
* <b>`regularizer`</b>: The regularizer is used to damped the stepsize when the
  iterations are becoming unstable. The bigger the regularizer is the
  smaller the stepsize becomes.
* <b>`regularizer_multiplier`</b>: If an iteration does not decrease the objective a
  new regularizer is computed by scaling it by this multiplier.
* <b>`callback`</b>: A callback function that will be called at each iteration. In
  graph mode the callback should return an op or list of ops that will
  execute the callback logic. The callback needs to be of the form
  f(iteration, objective_value, variables). A callback is a Python
  `callable`. The callback could be used for logging, for example if one
  wants to print the objective value at each iteration.
* <b>`name`</b>: A name for this op. Defaults to "levenberg_marquardt_minimize".


#### Returns:

The value of the objective function and variables attained at the final
iteration of the minimization procedure.



#### Raises:


* <b>`ValueError`</b>: If max_iterations is not at least 1.
* <b>`InvalidArgumentError`</b>: This exception is only raised in graph mode if the
Cholesky decomposition is not successful. One likely fix is to increase
the regularizer. In eager mode this exception is catched and the regularizer
is increased automatically.


#### Examples:


```python
x = tf.constant(np.random.random_sample(size=(1,2)), dtype=tf.float32)
y = tf.constant(np.random.random_sample(size=(3,1)), dtype=tf.float32)

def f1(x, y):
  return x + y

def f2(x, y):
  return x * y

def callback(iteration, objective_value, variables):
  def print_output(iteration, objective_value, *variables):
    print("Iteration:", iteration, "Objective Value:", objective_value)
    for variable in variables:
      print(variable)
  inp = [iteration, objective_value] + variables
  return tf.py_function(print_output, inp, [])

minimize_op = minimize(residuals=(f1, f2),
                       variables=(x, y),
                       max_iterations=10,
                       callback=callback)

if not tf.executing_eagerly():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(minimize_op)
```
