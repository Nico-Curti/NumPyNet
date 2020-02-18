# L2 normalization layer

The l2 normalizatioon layer normalizes the data with respect to the selected axis, using the l2  norm, computed as such:

![](https://latex.codecogs.com/gif.latex?||x||_2&space;=&space;\sqrt{\sum_{i=0}^N&space;x_i^2})

Where `N` is the dimension fo the selected axis. The normalization is computed as:

![](https://latex.codecogs.com/gif.latex?\hat&space;x&space;=&space;\frac{1}{\sqrt{\sum&space;x_i^2&space;&plus;&space;\epsilon}})

Where &epsilon; is a small (order of 10<sup>-8</sup>) constant used to avoid division by zero.

The backward, in this case, is computed as:

![](https://latex.codecogs.com/gif.latex?\delta^l&space;=&space;\delta^l&space;&plus;&space;\delta_{l-1}&space;&plus;&space;\frac{(1&space;-&space;\hat&space;x)}{\sqrt{\sum_i&space;x_i&space;&plus;&space;\epsilon}})

Where &delta;<sup>l</sup> is the delta to be backpropagated, while &delta;<sup>l-1</sup> is the delta coming from the next layer

This code is an example of how to use the single `l2norm_layer` object:

```python

import os

from NumPyNet.layers.l2norm_layer import L2Norm_layer

import numpy as np

# those functions rescale the pixel values [0,255]->[0,1] and [0,1->[0,255]
img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
inpt = np.asarray(Image.open(filename), dtype=float)
inpt.setflags(write=1)
inpt = img_2_float(inpt)

# add batch = 1
inpt = np.expand_dims(inpt, axis=0)

# instantiate the layer
layer = L2Norm_layer(axis=None) # axis=None just sum all the values

# FORWARD

layer.forward(inpt)
forward_out = layer.output # the shape of the output is the same as the one of the input

# BACKWARD

delta = np.zeros(shape=inpt.shape, dtype=float)
layer.backward(delta, copy=False)
```

To have a look more in details on what's happening, those are the definitions of `forward` and `backward` for the l2norm_layer:

```python
def forward(self, inpt):
  '''
  Forward of the l2norm layer, apply the l2 normalization over
  the input along the given axis
  Parameters:
    inpt: the input to be normaliza
  '''
  self._out_shape = inpt.shape

  norm = (inpt * inpt).sum(axis=self.axis, keepdims=True)
  norm = 1. / np.sqrt(norm + 1e-8)
  self.output = inpt * norm
  self.scales = (1. - self.output) * norm
  self.delta  = np.zeros(shape=self.out_shape, dtype=float)
```

That's just a simple implementation of the formualas described above:
  * sum of the input squared over the selected axis. If `self.axis` is `None` then the sum is computed considering every pixel.
  * `self.output` is `inpt` normalized
  * define `self.scale` and initialize `self.delta`

The `backward` is:

```python
def backward(self, delta, copy=False):
  '''
  Compute the backward of the l2norm layer
  Parameter:
    delta : global error to be backpropagated
  '''

  self.delta += self.scales
  delta[:]   += self.delta
```

which updates `self.delta` with the value of `self.scales` computed in `forward`, and then update the value of `delta`, received as argument from the function.
