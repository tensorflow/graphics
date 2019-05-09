# Debug Mode for TensorFlow Graphics

Tensorflow Graphics heavily relies on L2 normalized tensors, as well as
trigonometric functions that expect their inputs to be in a certain range.
During optimization, an update can make these variables take values that cause
these functions to return `Inf` or `NaN` values. To make debugging such issues
simpler, TensorFlow Graphics provides a debug flag that injects assertions to
the graph to check for the right ranges and the validity of the returned values.
As this can slow down the computations, debug flag is set to `False` by default.

Users can set the `-tfg_debug` flag to run their code in debug mode. The flag
can also be set programmatically by first importing these two modules:

```python
from absl import flags
from tensorflow_graphics.util import tfg_flags
```

and then by adding the following line to the code.

```python
flags.FLAGS[tfg_flags.TFG_ADD_ASSERTS_TO_GRAPH].value = True
```
