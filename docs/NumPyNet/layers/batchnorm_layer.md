### Batch Normalization Layer

Batch Normalization is the operation that involves the normalization of every feature (pixel) along the batch axis. If x<sub>i</sub> is the value of the x pixel in the i-th image of the batch, then $\bar x$:

$$
f(\bar x) = 6\\
f'(\bar x) = 0 \\
\mbox{some try with latex}
$$

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\bar&space;x&space;=\frac{1}{batch\_size}&space;\sum_{i=0}^{batch\_size}x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar&space;x&space;=\frac{1}{batch\_size}&space;\sum_{i=0}^{batch\_size}x_i" title="\bar x =\frac{1}{batch\_size} \sum_{i=0}^{batch\_size}x_i" /></a>
</p>

is the mean value of the pixel along the batch axis.
The layer is being implemented following the original paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), where is also possible to find the entire implementation of the algorithm and teoretical explanations of forward and backward pass.

# IMAGE


In the code below we present an example on how to use this single layer as it is:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer # class import

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3) # batch != 1 in this case
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import some images from file

# scales (gamma) and bias (Beta) initialization
scales = np.random.uniform(low=0., high=1., size=(w, h, c))
bias   = np.random.uniform(low=0., high=1., size=(w, h, c))

# Layer initialization, with parameters scales and bias
layer = Avgpool_layer(scales=scales, bias=bias)

# Forward pass
layer.forward(inpt=input, copy=False)
out_img = layer.output    # the output in this case will be of shape=(batch, w, h, c), so a batch of normalized images


# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)     # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified and ready to be passed to the previous layer.delta
```
