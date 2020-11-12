# Convolutional Layer

Convolutional Neural Network (CNN) are designed in particular for image analysis.
Convolution is the mathematical integration of two functions in which the second one is translated by a given value:

![](https://latex.codecogs.com/gif.latex?(f&space;*&space;g)(t)&space;=&space;\int_{-\infty}^{&plus;\infty}&space;f(\tau)g(t&space;-&space;\tau)d\tau)

In signal processing this operation is also called *crossing correlation* and it is equivalent to the *autocorrelation* function computed in a given point.
In image processing the first function is represented by the image *I* and the second one is a kernel *k* (or filter) which shift along the image.
In this case we will have a 2D discrete version of the formula given by:

![](https://latex.codecogs.com/gif.latex?C&space;=&space;k&space;*&space;I)

![](https://latex.codecogs.com/gif.latex?C[i,&space;j]&space;=&space;\sum_{u=-N}^{N}&space;\sum_{v=-M}^{M}&space;k[u,&space;v]&space;\cdot&space;I[i&space;-&space;u,&space;j&space;-&space;v])

where `C[i, j]` is the pixel value of the resulting image and `N, M` are kernel dimensions.

The use of CNN in modern image analysis applications can be traced back to multiple causes.
First of all the image dimensions are increasingly bigger and thus the number of variables/features, i.e pixels, is often too big to manage with standard DNN.
Moreover if we consider detection problems, i.e the problem of detecting an set of features (or an object) inside a larger pattern, we want a system able to recognize the object regardless of where it appears into the input.
In other words, we want that our model would be independent by simple translations.

Both the above problems can be overcome by CNN models using a small kernel, i.e weight mask, which maps the full input.
A CNN is able to successfully capture the spatial and temporal dependencies in any signal through the application of relevant filters.

In the image below some example on how the application of different kernel can highlight different features.

Here's an example code on how to use the single convolutional layer:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.convolutional_layer import Convolutional_layer # class import
from NumPyNet import activations

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3) # batch != 1 in this case
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import some images from file

out_channels = num_filters = 10    # number of channels of the output image

filters      = np.random.uniform(low=0., high=1., size=(size, size, c, out_channels))
bias         = np.random.uniform(low=0., high=1., size=(out_channels,))

activ_func = activations.Relu() # it can also be:
                                #     activations.Relu  (class Relu)
                                #     "Relu" (a string, case insensitive)

# Layer initialization, with parameters scales and bias
layer = Convolutional_layer(input_shape=input.shape,     # shape of the input, batch included
                            filters=out_channels,        # number of filter to apply
                            weights=filters,             # filters to be applied
                            bias=bias,
                            activation=layer_activation, # activation function
                            size=size,                   # size of the kernel
                            stride=stride,               # stride of the kernel
                            pad=pad)                     # padding (boolean)


# Forward pass
layer.forward(inpt=input, copy=False)
out_img = layer.output    # the output in this case will be a batch of images of shape = (batch, out_width, out_heigth , out_channels)

# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)     # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified and ready to be passed to the previous layer.delta
# and also the updates for weights and bias are computed in the backward

# update of the trainable weights (filters and bias)
layer.update(momentum=0., decay=0., lr=1e-2, lr_decay=1.)

```

To have a look more in details on what's happening, here's the definition of `forward` and `backward` function:


```python
def forward(self, inpt, copy=False):
  '''
  Forward function of the Convolutional Layer: it convolves an image with 'channels_out'
    filters with dimension (kx,ky, channels_in). In doing so, it creates a view of the image
    with shape (batch, out_w, out_h, in_c, kx, ky) in order to perform a single matrix
    multiplication with the reshaped filters array, which shape is (in_c * kx * ky, out_c).

  Parameters
  ----------
    inpt : array-like
      input batch of images in format (batch, in_w, in_h, in _c)

    copy : boolean (default=False).
      If False the activation function modifies its input, if True make a copy instead

  Returns
  -------
    self
  '''

  self._check_dims(shape=self.input_shape, arr=inpt, func='Forward')

  kx, ky = self.size
  sx, sy = self.stride
  _, w, h, _ = self.input_shape
  inpt = inpt.astype('float64')

  # Padding
  if self.pad:
    mat_pad = self._pad(inpt)
  else:
    # If no pad, every image in the batch is cut
    mat_pad = inpt[:, : (w - kx) // sx * sx + kx, : (h - ky) // sy * sy + ky, ...]

  # Create the view of the array with shape (batch, out_w ,out_h, kx, ky, in_c)
  self.view = self._asStride(mat_pad)

  # the choice of numpy.einsum is due to reshape of self.view being a copy
  z = np.einsum('lmnijk, ijko -> lmno', self.view, self.weights, optimize=True) + self.bias

  # (batch, out_w, out_h, out_c)
  self.output = self.activation(z, copy=copy)

  self.delta = np.zeros(shape=self.out_shape, dtype=float)

  return self
```

Those are the steps of the computation:

  * Padding of the images: if `self.pad` is `True`, then `_evaluate_padding` computes the number of rows/columns of Zeros added to every image in the batch, while `_pad` is just a wrap for :
  ```python
  numpy.pad(array=inpt,
            pad_width=((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
            mode='constant',
            constant_values=(0., 0.))
  ```
  if pad is false images are cut.

  * Similarly to Pool layers, here the method `_asStride` create a **view**  of the padded image with shapes `(batch, out_width, out_height, size, size, channels)`, that contains every size * size matrix of the image as if the kernel strided on it.
  * The following line is where the convolution takes place: the function [einsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html) is told to sum the last 3 axis of `self.view` and the first 3 axis of `self.weights` (aka filters) to output an array of shape `(batch, out_width, out_height, out_channels)`. We chose this function because every possible reshape didn't return a **view** of the original array.
  * add bias and activate the output.

The Backward function instead:

```python
def backward(self, delta, copy=False):
  '''
  Backward function of the Convolutional layer. Source:
  https://arxiv.org/abs/1603.07285

  Parameters
  ----------
    delta : array-like
      delta array of shape (batch, w, h, c). Global delta to be backpropagated.

    copy : bool (default=False)
      States if the activation function have to return a copy of the input or not.

  Returns
  -------
    self
  '''

  check_is_fitted(self, 'delta')
  self._check_dims(shape=self.input_shape, arr=delta, func='Backward')
  delta[:] = delta.astype('float64')

  self.delta *= self.gradient(self.output, copy=copy)

  self.weights_update = np.einsum('ijklmn, ijko -> lmno', self.view, self.delta, optimize=True)
  self.bias_update = self.delta.sum(axis=(0, 1, 2))  # shape = (channels_out)

  # Rotated weights, as theory suggest
  w_rot = np.rot90(self.weights, 2, axes=(0, 1))

  # Pad and dilate the delta array, then stride it and convolve
  self.delta = self._dilate_pad(self.delta)
  delta_view = self._asStride(self.delta, back=True)

  delta[:] = np.einsum('ijklmn, lmon -> ijko', delta_view, w_rot, optimize=True)

  return self
```

Which computes &delta;W, &delta;&beta; and the error &delta; to be backpropagated through a Deconvolution operation (or transposed convolution) of &delta; with the rotated weights matrix. The aritmetics (input and output dimension) of the operations is described in details in [this article](https://arxiv.org/pdf/1603.07285.pdf).
