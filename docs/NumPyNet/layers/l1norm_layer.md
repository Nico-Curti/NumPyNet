### L1 normalization Layer

The l1 normalizatioon layer normalizes the data with respect to the selected axis, using the l1 norm, computed as such:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=||x||_1&space;=&space;\sum_{i=0}^{N}&space;|x_i|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?||x||_1&space;=&space;\sum_{i=0}^{N}&space;|x_i|" title="||x||_1 = \sum_{i=0}^{N} |x_i|" /></a>
</p>

Where `N` is the dimension of the selected axis. The normalization is computed as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;x&space;=&space;\frac{x}{||x||_1&space;&plus;&space;\epsilon}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;x&space;=&space;\frac{x}{||x||_1&space;&plus;&space;\epsilon}" title="\hat x = \frac{x}{||x||_1 + \epsilon}" /></a>
</p>

Where &epsilon; is a small (order of 10<sup>-8</sup>) constant used to avoid division by zero.

The backward, in this case, is computed as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\delta^l&space;=&space;\delta^l&space;&plus;&space;\delta^{l-1}&space;-&space;sgn(\hat&space;x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta^l&space;=&space;\delta^l&space;&plus;&space;\delta^{l-1}&space;-&space;sgn(\hat&space;x)" title="\delta^l = \delta^l + \delta^{l-1} - sgn(\hat x)" /></a>
</p>

Where &delta;<sup>l</sup> is the previous layer's delta, and &delta;<sup>l-1</sup> is the next layer delta.

The code below is an example on how to use the single layer:
```python
import os

from NumPyNet.layers.l1norm_layer import L1Norm_layer

import numpy as np 

# those functions rescale the pixel values [0,255]->[0,1] and [0,1->[0,255]
img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
inpt = np.asarray(Image.open(filename), dtype=float)
inpt.setflags(write=1)
inpt = img_2_float(inpt) # preparation of the image

# add batch = 1
inpt = np.expand_dims(inpt, axis=0)

# instantiate the layer
layer = L1Norm_layer(axis=None) # axis=None just sum all the values

# FORWARD

layer.forward(inpt)
forward_out = layer.output # the shape of the output is the same as the one of the input

# BACKWARD

delta = np.zeros(shape=inpt.shape, dtype=float)
layer.backward(delta, copy=False)
```

To have a look more in details on what's happening, here are presented the defition of the functions `forward` and `backward`:

```python
def forward(self, inpt):
  '''
  Forward of the l1norm layer, apply the l1 normalization over
  the input along the given axis
  Parameters:
    inpt: the input to be normaliza
  '''
  self._out_shape = inpt.shape

  norm = np.abs(inpt).sum(axis=self.axis, keepdims=True)
  norm = 1. / (norm + 1e-8)
  self.output = inpt * norm
  self.scales = -np.sign(self.output)
  self.delta  = np.zeros(shape=self.out_shape, dtype=float)
```
The `forward` function is an implemenatation of what's stated before:
  * compute the inverse of the L1 norm, over the axis selected during the initialization of the layer objec. If `self.axis` is `None`, then the sum counts every pixels
  * compute `self.output` with the formula previuosly described
  * instantiate `self.scale`, used in `backward`

```python
def backward(self, delta, copy=False):
  '''
  Compute the backward of the l1norm layer
  Parameter:
    delta : global error to be backpropagated
  '''

  self.delta += self.scales
  delta[:]   += self.delta
```
As for `forward`, `backward` is just a simple implementation of what's described above.
