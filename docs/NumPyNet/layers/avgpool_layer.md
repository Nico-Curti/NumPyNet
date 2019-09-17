### Average Pool Layer

The Average Pool layer perfoms a down sampling on the batch of images.
It slides a 2D kernel of arbitrary size over the image and the output is the mean value of the pixels inside the kernel. The slide of the kernel (how many pixel it moves on the x and the y axis) can be modified.

In the images below are shown some results obtained by performing an average pool (forward and backward) with different kernel sizes and strides:


<p align="center">
  <img src="https://github.com/Nico-Curti/NumPyNet/blob/master/docs/NumPyNet/images/average_3-2.png" >
</p>
<p align="center">
  <img src="https://github.com/Nico-Curti/NumPyNet/blob/master/docs/NumPyNet/images/average_30-20.png">
</p>
*Fig.1: in the image are shown the effects of different kernel size-stride couplets. From up to down : size=3 and stride=2, size=30 and stride=20*

(I'm not showing the backward image, since it looks like a random noise)
The code used to obtain those images can be found [in this repository](https://github.com/Nico-Curti/NumPyNet/blob/master/NumPyNet/layers/avgpool_layer.py), after the average pool layer class definition.

This is an example code on how to use the single layer to perform its `forward` and `backward` functions:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.avgpool_layer import Avgpool_layer # class import

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3)
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import an image from file

size   = 3 # definition of size and stride of the kernel
stride = 2
pad    = False # it's possible to pad the image (used to not lose information arounde image edges.)

# Layer initialization
layer = Avgpool_layer(size=size, stride=stride, pad=pad)

# Forward pass
layer.forward(inpt=input, copy=False)
out_img = layer.output    # the output in this case will be of shape=(batch, out_w, out_h, c), so a batch of images


# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)     # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified and ready to be passed to the previous layer.delta
```

To have a look more in details on what's happening:

##### Forward:

```python
def forward(self, inpt):
  '''
  Forward function of the average pool layer: it slide a kernel of size (kx,ky) = size
  and with step (st1, st2) = strides over every image in the batch. For every sub-matrix
  it computes the average value without considering NAN value (padding), and passes it
  to the output.

  Parameters:
    inpt : input batch of image, with the shape (batch, input_w, input_h, input_c)
  '''

  self.batch, self.w, self.h, self.c = inpt.shape
  kx, ky = self.size
  sx, sy = self.stride

  # Padding
  if self.pad:
    self._evaluate_padding()
    mat_pad = self._pad(inpt)
  else:
    # If padding false, it cuts images' raws/columns
    mat_pad = inpt[:, : (self.w - kx) // sx*sx + kx, : (self.h - ky) // sy*sy + ky, ...]

  # 'view' is the strided input image, shape = (batch, out_w, out_h, out_c, kx, ky)
  view = self._asStride(mat_pad, self.size, self.stride)

  # Mean of every sub matrix, computed without considering the pad (np.nan)
  self.output = np.nanmean(view, axis=(4, 5))
  self.delta  = np.zeros(shape=self.out_shape, dtype=float)
```

In the first place, if required by the user, the image is padded:

  1. The function `_evaluate_padding` take no parameters and compute the number of rows/columns to be added to every image on the batch, following keras SAME padding as described [here](https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave).
  2. The function `_pad` is just a wrap for:

```python
numpy.pad(array=inpt, pad_with=((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)), mode='constant', constant_values=(np.nan, np.nan))
```
that pads the images with a number of rows equal to pad_top + pad_bottom, and a number of columns equal to pad_left + pad_right. All values are np.nan.

  3. If no padding is requested, the colums and rows that would be left out from the kernel sliding are cut from every image on the batch.

Then the padded images are passed as argument to `_asStride`, that returns a **view** of the strided image. A view contains the same data as the original array, but arranged in a different way, without taking up more space.

The variable `view` stores data in the shapes (batch, out_width, out_height, channels, size, size):
basically N = batch * out_width * out_height * c matrices size * size, or every set of pixels under the kernel slice.

The output dimensions of the image are computed as such:

  <a href="https://www.codecogs.com/   eqnedit.php?latex=out\_width&space;=&space;\lfloor\frac{width&space;&plus;&space;pad&space;-&space;size}{stride}\rfloor&space;&plus;&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?out\_width&space;=&space;\lfloor\frac{width&space;&plus;&space;pad&space;-&space;size}{stride}\rfloor&space;&plus;&space;1" title="out\_width = \lfloor\frac{width + pad - size}{stride}\rfloor + 1" /></a>

  <a href="https://www.codecogs.com/eqnedit.php?latex=out\_height&space;=&space;\lfloor\frac{height&space;&plus;&space;pad&space;-&space;size}{stride}\rfloor&space;&plus;&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?out\_height&space;=&space;\lfloor\frac{height&space;&plus;&space;pad&space;-&space;size}{stride}\rfloor&space;&plus;&space;1" title="out\_height = \lfloor\frac{height + pad - size}{stride}\rfloor + 1" /></a>

After the view is created, the output is computed by `numpy.nanmean(view, axis=(4,5))`, which returns an array containing all the N average values, excluding nan value from the equations.

The next code shows, instead, the backward function definition :

```python

def backward(self, delta):
  '''
  backward function of the average_pool layer: the function modifies the net delta
  to be backpropagated.
  Parameters:
    delta : global delta to be backpropagated with shape (batch, out_w, out_h, out_c)
  '''

  kx, ky = self.size

  # Padding delta for a coherent _asStride dimension
  if self.pad:
    mat_pad = self._pad(delta)
  else :
    mat_pad = delta

  # _asStride of padded delta let me access every pixel of the memory in the order I want.
  # This is used to create a 1-1 correspondence between output and input pixels.
  net_delta_view = self._asStride(mat_pad, self.size, self.stride)

  # needs to count only non nan values for keras
  b, w, h, c = self.output.shape
  combo = itertools.product(range(b), range(w), range(h), range(c)) # every combination of indices

  # The indexes are necessary to access every pixel value one at a time, since
  # modifing the same memory address more times at once doesn't produce the correct result

  for b, i, j, k in combo:
    norm = 1./ np.count_nonzero(~np.isnan(net_delta_view[b, i, j, k, :])) # this only counts non nan values for norm
    net_delta_view[b, i, j, k, :] += self.delta[b, i, j, k] * norm

  # Here delta is updated correctly
  if self.pad:
    _, w_pad, h_pad, _ = mat_pad.shape
    # Excluding the padded part of the image
    delta[:] = mat_pad[:, self.pad_top : w_pad - self.pad_bottom, self.pad_left : h_pad - self.pad_right, :]
  else:
    delta[:] = mat_pad
```

The backpropagation of the average pool layer implies that every value of `delta` is updated with the corresponding value of `layer.delta` multiplied by a normalization factor, that is the inverse of the number of non nan values in a kernel windows.

Unfortunately, the use of the `view` object is not "thread safe", so if there are superposition between two kernel windows (stride < size), the same pixel will be present more than one time in `net_delta_view`. That requires a "one a time" modification of every single pixels.

In the end of the function delta is modified with the correct values of `mat_pad`, excluding nan values added by padding.
