### Input Layer

This layer is more of an utility: it's the first layer for every `network` object and all it does is passing the exact same input that it receives, checking that `input_shape` is consistent with the effective input shape.
Even the backward just pass back the delta it receive from the next layer.

To use this layer as a single layer, this is a simple example code:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.input_layer import Input_layer # class import

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all
# the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3)
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import an image from file

# Layer initialization
layer = Input_layer(input_shape=input.shape)

# Forward pass
layer.forward(inpt=input)
out_img = layer.output    # the output in this case will be of shape=(batch, w, h, c), so a batch of images (identical to the input actually)

# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)     # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified
```

To have a look more in details on what's happening, those are the definitions of `forward` and `backward` functions for this layer:

```python
def forward(self, inpt):
  '''
  Simply store the input array
  Parameters:
    inpt: the input array
  '''
  if self.out_shape != inpt.shape:
    raise ValueError('Forward Input layer. Incorrect input shape. Expected {} and given {}'.format(self.out_shape, inpt.shape))

  self.output[:] = inpt
  self.delta = np.zeros(shape=self.out_shape, dtype=float)
```

As stated above, all it does is check that the input shape is consistent with `self.out_shape`, that is the same as `input_shape`
And here's the backward:

```python
def backward(self, delta):
  '''
  Simply pass the gradient
  Parameter:
    delta : global error to be backpropagated
  '''
  if self.out_shape != delta.shape:
    raise ValueError('Forward Input layer. Incorrect delta shape. Expected {} and given {}'.format(self.out_shape, delta.shape))

  delta[:] = self.delta
```
That does nothing more than updating `delta` with `layer.delta` exactly as it is.
