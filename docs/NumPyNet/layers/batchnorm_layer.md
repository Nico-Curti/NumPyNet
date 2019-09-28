### Batch Normalization Layer

Batch Normalization is the operation that involves the normalization of every feature (pixel) along the batch axis.

The layer has been implemented following the original paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167), where is also possible to find the entire implementation of the algorithm and teoretical explanations of forward and backward pass.

According to the paper, if x_i is the value of the x pixel in the i-th image of the batch, where i range from 1 to `batch_size`, then the forward pass look as follow:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\bar&space;x&space;=\frac{1}{batch\_size}&space;\sum_{i=0}^{batch\_size}x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bar&space;x&space;=\frac{1}{batch\_size}&space;\sum_{i=0}^{batch\_size}x_i" title="\bar x =\frac{1}{batch\_size} \sum_{i=0}^{batch\_size}x_i" /></a>
</p>
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma^2&space;=&space;\frac{1}{batch\_size}\sum_{i=1}^{batch\_size}(x_i-\bar&space;x)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma^2&space;=&space;\frac{1}{batch\_size}\sum_{i=1}^{batch\_size}(x_i-\bar&space;x)^2" title="\sigma^2 = \frac{1}{batch\_size}\sum_{i=1}^{batch\_size}(x_i-\bar x)^2" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;x_i&space;=&space;\frac{(x_i&space;-\bar&space;x)^2}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;x_i&space;=&space;\frac{(x_i&space;-\bar&space;x)^2}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}" title="\hat x_i = \frac{(x_i -\bar x)^2}{\sqrt{\sigma^2 + \epsilon}}" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y_i&space;=&space;\gamma&space;\hat&space;x_i&space;&plus;&space;\beta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i&space;=&space;\gamma&space;\hat&space;x_i&space;&plus;&space;\beta" title="y_i = \gamma \hat x_i + \beta" /></a>
</p>

Where y_i is the x pixel normalized, rescaled and shifted of the i-th image in the batch, and &epsilon; is a (very small) constant, to avoid division by zero.

On the other hand, the backward is slightly more complicated, since we have to derive some formulas for the backpropagation and, moreover, &gamma; (scales) and &beta;'s (biases) update values (&delta;&gamma; and &delta;&beta; ), because they're trainable weigths.

The two updates are computed as:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;\beta&space;=&space;\sum_{i=0}^{batch\_size}&space;\delta_i^l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;\beta&space;=&space;\sum_{i=0}^{batch\_size}&space;\delta_i^l" title="\delta \beta = \sum_{i=0}^{batch\_size} \delta_i^l" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;\gamma&space;=&space;\sum_{i=0}^{batch\_size}&space;\delta_i^l&space;\cdot&space;\hat&space;x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;\gamma&space;=&space;\sum_{i=0}^{batch\_size}&space;\delta_i^l&space;\cdot&space;\hat&space;x_i" title="\delta \gamma = \sum_{i=0}^{batch\_size} \delta_i^l \cdot \hat x_i" /></a>
<p>

Where &delta;l is the error passed from the next layer.
And the formula for the error to be back-propaate &delta;l-1 is :

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\delta^{l-1}_i&space;=&space;\frac{batch\_size&space;\cdot&space;\delta&space;\hat&space;x_i&space;-&space;\sum_{j=0}^{batch\_size}\delta&space;\hat&space;x_j&space;-&space;\hat&space;x_i&space;\cdot&space;\sum_{j=0}^{batch\_size}\delta&space;\hat&space;x_j&space;\cdot&space;\hat&space;x_j}{batch\_size&space;\cdot&space;\sqrt{\sigma^2&space;&plus;&space;\epsilon}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta^{l-1}_i&space;=&space;\frac{batch\_size&space;\cdot&space;\delta&space;\hat&space;x_i&space;-&space;\sum_{j=0}^{batch\_size}\delta&space;\hat&space;x_j&space;-&space;\hat&space;x_i&space;\cdot&space;\sum_{j=0}^{batch\_size}\delta&space;\hat&space;x_j&space;\cdot&space;\hat&space;x_j}{batch\_size&space;\cdot&space;\sqrt{\sigma^2&space;&plus;&space;\epsilon}}" title="\delta^{l-1}_i = \frac{batch\_size \cdot \delta \hat x_i - \sum_{j=0}^{batch\_size}\delta \hat x_j - \hat x_i \cdot \sum_{j=0}^{batch\_size}\delta \hat x_j \cdot \hat x_j}{batch\_size \cdot \sqrt{\sigma^2 + \epsilon}}" /></a>
</p>

Where &delta;&gamma;, &delta;&beta; etc... are the derivatives of `y` with the correspondent variable.

For an in details derivation, check the paper we linked above or [this very clear blog](https://kevinzakka.github.io/2016/09/14/batch_normalization).

In the code below we present an example on how to use this single layer as it is:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer # class import

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3) # batch != 1 in this case
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import some images from file

# scales (gamma) and bias (Beta) initialization (trainable weights!)
scales = np.random.uniform(low=0., high=1., size=(w, h, c))
bias   = np.random.uniform(low=0., high=1., size=(w, h, c))

# Layer initialization, with parameters scales and bias
layer = BatchNorm_layer(scales=scales, bias=bias)

# Forward pass
layer.forward(inpt=input, copy=False)
out_img = layer.output    # the output in this case will be of shape=(batch, w, h, c), so a batch of normalized, rescaled and shifted images


# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)     # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified and ready to be passed to the previous layer.delta
# and also the updates for scales and bias are computed

# update of the trainable weights
layer.update(momentum=0., decay=0., lr=1e-2, lr_decay=1.)

```

To have a look more in details on what's going on in the forward and backward pass:

```python
def forward(self, inpt, epsil=1e-8):
  '''
  Forward function of the BatchNormalization layer. It computes the output of
  the layer, the formula is :
                  output = scale * input_norm + bias
  Where input_norm is:
                  input_norm = (input - mean) / sqrt(var + epsil)
  where mean and var are the mean and the variance of the input batch of
  images computed over the first axis (batch)
  Parameters:
    inpt  : numpy array, batch of input images in the format (batch, w, h, c)
    epsil : float, used to avoi division by zero when computing 1. / var
  '''

  self._out_shape = inpt.shape

  # Copy input, compute mean and inverse variance with respect the batch axis
  self.x    = inpt
  self.mean = self.x.mean(axis=0)  # Shape = (w, h, c)
  self.var  = 1. / np.sqrt((self.x.var(axis=0)) + epsil) # shape = (w, h, c)
  # epsil is used to avoid divisions by zero

  # Compute the normalized input
  self.x_norm = (self.x - self.mean) * self.var # shape (batch, w, h, c)
  self.output = self.x_norm.copy() # made a copy to store x_norm, used in Backward

  # Output = scale * x_norm + bias
  if self.scales is not None:
    self.output *= self.scales  # Multiplication for scales

  if self.bias is not None:
    self.output += self.bias # Add bias

  # output_shape = (batch, w, h, c)
  self.delta = np.zeros(shape=self.out_shape, dtype=float)
```

The forward is basically a numpy version of the theory described above:

  * compute the mean over the batch axis for every pixel.
  * compute the inverse square root of the variance on the same axis.
  * compute the normalized input.
  * multiply for scales and add bias if necessary.

Here is the code for the backward function :


```python
def backward(self, delta=None):
  '''
  BackPropagation function of the BatchNormalization layer. Every formula is a derivative
  computed by chain rules: dbeta = derivative of output w.r.t. bias, dgamma = derivative of
  output w.r.t. scales etc...
  Parameters:
    delta : the global error to be backpropagated, its shape should be the same
      as the input of the forward function (batch, w, h ,c)
  '''

  invN = 1. / np.prod(self.mean.shape)

  # Those are the explicit computation of every derivative involved in BackPropagation
  # of the batchNorm layer, where dbeta = dout / dbeta, dgamma = dout / dgamma etc...

  self.bias_updates   = self.delta.sum(axis=0)                 # dbeta
  self.scales_updates = (self.delta * self.x_norm).sum(axis=0) # dgamma

  self.delta *= self.scales                                    # self.delta = dx_norm from now on

  self.mean_delta = (self.delta * (-self.var)).mean(axis=0)    # dmu
  self.var_delta  = ((self.delta * (self.x - self.mean)).sum(axis=0) *
                   (-.5 * self.var * self.var * self.var))     # dvar

  # Here, delta is the derivative of the output w.r.t. input
  self.delta = (self.delta * self.var +
                self.var_delta * 2 * (self.x - self.mean) * invN +
                self.mean_delta * invN)

  if delta is not None:
    delta[:] = self.delta
```

Here every single step of the derivation is computed singularly:

  * compute bias and scales updates as described above
  * <a href="https://www.codecogs.com/eqnedit.php?latex=\delta&space;\hat&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta&space;\hat&space;x" title="\delta \hat x" /></a> is computed modifying directly the variable &delta;
  * then with the derivatives w. r. t. the mean and to the variance are used to compute the delta to be backpropagated
