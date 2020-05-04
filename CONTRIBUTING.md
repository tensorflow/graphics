# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## What features to add?

The library is open to any contributions along the lines of computer graphics,
with the top-level themes being rendering, physics simulation, and geometry
processing. Contributions can be in the form of low level functions (majority of
the library), Neural Networks layers or Colaboratory notebooks.

## Guidelines for Tensorflow operations

TensorFlow Graphics follows the TensorFlow
[contributor guidelines](https://www.tensorflow.org/community/contribute) and
[code style guide](\(https://www.tensorflow.org/community/contribute/code_style\)).
Besides these, TensorFlow Graphics has a few more guidelines which you can find
below.

### Programming languages

Unless this comes at a significant performance hit, pure Python is preferred.

### Structure of a function

The general structure of a function should be as follows:

*   Name of the function followed by inputs to that function
*   Doc-string documentation
*   Definition of the scope using `tf.compat.v1.name_scope`
*   Functions that take tensors as arguments should call `tf.convert_to_tensor`
*   Checking the shape and value of the inputs as necessary
*   Main logic of the function

### Function names

Prefer function names that are concise, descriptive, and integrate well with the
module name when called. For instance, the `rotate` function from the
`rotation_matrix_3d` sub-module can be called using `rotation_matrix_3d.rotate`,
and makes it easy for anyone to understand what is being calculated. Functions
that are only meant to be local to the file in which they are written should
have an underscore before their name.

### Input parameters

The first arguments should be tensors, followed by python parameters, and
finally the name scope for the TensorFlow operation.

### Input shapes

*   The first dimensions of a tensor should represent the shape of the batch,
    and the last dimensions should represent the core shape of the elements used
    by the function. For instance, `rotation_matrix_3d.rotate` accepts rotation
    matrices of shape `[A1, ..., An, 3, 3]` where `[A1, ..., An]` are the
    optional batch dimensions, and `[3, 3]` is the shape required to capture 3D
    rotation matrices.
*   Every function must support batch dimensions of any shape, including tensors
    with no batch dimensions.
*   For input tensors with common batch shapes, document whether they can be
    broadcast compatible or not, and try to make them compatible when possible
    by, for instance, using `shape.get_broadcasted_shape` and `tf.broadcast_to`.

### Documentation

Every function must have a docstring-type documentation describing what the
function is performing, its arguments, and what is returned. The input sizes
must be written between backquotes with batch dimensions indexed by letters and
numbers, for instance: \`[A1, ..., An, 3]\`. Here `[A1, ..., An]` are the batch
dimensions, and 3 is the intrinsic dimension required for the operation (e.g. a
point in 3d). Prefer to put the batch dimension first.

### Error handling

Handling unexpected inputs usually consists in checking that their shapes are
consistent with expectations, which can be performed with `shape.check_static`,
but also checking that the content of the tensors are valid (e.g. value in a
specific range, no NaNs etc.), which can be performed with utilities provided in
the `asserts` module.

### Differentiability and stable gradients

There are several TF operations that can turn derivatives to zero at unintended
points of your functions / operations. This can be avoided by using tools
provided in the util.safe_ops module. If it can not be avoided, make sure to add
tests checking the Jacobians of the function at the potentially discontinuous
points of the function. See [Testing Jacobians](#testing-jacobians) below.
Examples of such functions include:

*   tf.maximum / tf.minimum(a(x), b(x)): These create piecewise functions, which
    means derivatives can be discontinuous or zero for some ranges or points.
*   tf.clip_by_value / tf.clip_by_norm: These are also piecewise functions where
    the actual function is replaced with a constant piece for certain points or
    ranges, which makes the derivative zero, even if it actually isn’t.
*   tf.where(cond, a, b): This is another way of creating piecewise functions.
    This should be used only if it is really meant to create a piecewise
    function.

The util.safe_ops submodule contains helper functions that can resolve issues
with divisions by zero, but also helpers to ensure that the data is in the
appropriate range. For instance a dot product of two normalized vectors can
result in values outside of [-1.0, 1.0] range due to fixed point arithmetic.
This in turn may result in NaN if used with arcsin or arccos. In such cases,
safe_shrink in util.safe_ops should be used rather than clipping the range,
since clipping removes derivatives which should be non-zero at these points.
Cases involving zero divided by zero are a bit more involved and require
dedicated workarounds.

### Software compatibility

The library is intended to be compatible with the latest stable TensorFlow 1
release as well as the latest nightly package for TensorFlow 2. We also aim to
be compatible with a couple of versions of Python. Testing for all the above is
automatically performed using
[travis](https://travis-ci.org/tensorflow/graphics).

### Hardware compatibility

Except for performance reasons, every function must be hardware agnostic (e.g.
CPU / GPU / TPU).

### Python modules

Each module must contain a \_\_init__.py file which lists all the sub-modules it
contains.

## Tests

Testing code is essential to make the library usable by everyone at all times.
In the following, we will briefly review our policies around unit testing and
code coverage.

### Unit testing

*   all test classes must derive from
    tensorflow_graphics.util.test_case.TestCase
*   to improve readability of the code, and minimize duplication, the parameters
    passed to all the test functions described below are passed using
    `parameterized.parameters` provided by `absl.testing`.

#### Test files

Each module containing code has associated tests in the module's test
sub-folder. Each test sub-folder must contain an empty \_\_init__.py, and one
file per .py file in the module. For instance, if the `transformation` module
contains `quaternion.py`, the tests associated with that python file should be
located in `transformation/tests/quaterion_test.py`.

In the following, we use FN as shorthand for the name of the function to be
tested. Let's now have a look at how tests are structured and specific things to
test for.

#### Structure of a test

TensorFlow Graphics follow the arrange-act-assert testing pattern. Moreover, if
multiple tests are used in a single function to test for different but similar
behavior, self.subTest should be used to create separate blocks.

#### Testing return values

The function names and behavior to use for testing return values are as follows:

*   `test_FN_random` to ensure that functions return the expected result for any
    valid input.
*   `test_FN_preset` to test specific inputs, and to make sure that corner cases
    are handled appropriately.

#### Error handling

Following are the function names and behavior to use for testing that errors are
handled appropriately:

*   `test_FN_exception_raised` to test that functions return the expected error
    messages when input parameters are invalid (e.g. shape or values).
*   `test_FN_exception_not_raised` to make sure that valid arguments do not
    raise any errors.

N.B.: For both test functions above, make sure to include `None` in some of the
input shapes.

#### Testing Jacobians

Derivatives and gradients being at the core of Deep Learning training
algorithms, testing for the stability and correctness of gradients is core to
prevent problems, especially while training large networks. We perform numerical
differentiation to ensure the correctness and stability of the Jacobians of any
function by defining:

*   `test_FN_jacobian_random` to ensure that Jacobians are correct on the whole
    input domain.

*   `test_FN_jacobian_preset` to test the stability of Jacobian around corner
    cases, or points where the function might not be smooth / continuous.

N.B.: for both test functions above, make sure to decorate them with
`@flagsaver.flagsaver(tfg_add_asserts_to_graph=False)` to avoid potential errors
arising due to finite differentiation (e.g. tensor not normalized anymore)

### Coverage

The GitHub mirror of Tensorflow Graphics is using
<a href="https://coveralls.io/">coveralls</a> to assess the test coverage. The
version of Tensorflow Graphics that is internal to Google contains the same
features compared to what is available on GitHub, but has access to more tools
for testing. For this project, our internal policy is to only submit code for
which our internal testing tools report at least 99% coverage. This number might
seem to be a steep requirement, but given the nature of the project, this is
obtained with reasonable efforts.
