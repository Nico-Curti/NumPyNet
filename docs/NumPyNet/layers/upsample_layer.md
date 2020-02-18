# Upsample Layer

To feed a super-resolution model we have to use a series of prior-known LR-HR image association; starting from this considerations we can down-sample our images by a desired scale factor: tipically between 2 and 8.
On the other hand, thought, we can also consider a set of images as the LR ones, and obtain the
Thhese feats can be achieved using a Pooling algorithm (in particular an average [AveragePooling](./avgpool_layer.md)) for the down-sample or we can use an UpSample layer.

The UpSample function is commonly related to GAN (Generative Adversarial Networks) models in which we have to provide a series of artificial images to a given Neural Network, but it's also a function that can be introduced inside a Neural Network model to rescale the number of features.
The UpSample function inside a Neural Network model has to provide both up- and down- sampling technique since one is used in the `forward` function, while its inverse during the back-propagation.

This is an example code on how to use the sngle UpSample layer:

```python
from NumPyNet.layers.upsample_layer import Upsample_layer

import numpy as np # the library is entirely based on numpy

batch, w, h, c = (5, 100, 100, 3)
inpt = np.random.uniform(low=0., high=1., size(batch, w, h, c))

stride = -3 # in this case we will have a downsample by factor 3 x 3 = 9
scale = 1.5

layer = Upsample_layer(scale=scale, stride=stride)

# FORWARD

layer.forward(inpt)
forward_out = layer.output # donwscaled images
print(layer)

# BACKWARD

layer.delta = layer.output
delta = np.empty(shape=inpt.shape, dtype=float)
layer.backward(delta)

# now delta is updated and ready to be backpropagated.
```

To have a look more in details on what's happening, this is the definition of `forward`:

```python
def forward(self, inpt):
	'''
	Forward of the upsample layer, apply a bilinear upsample/downsample to
	the input according to the sign of stride

	Parameters:
		inpt: the input to be up-down sampled
	'''
	self.batch, self.w, self.h, self.c = inpt.shape

	if self.reverse: # Downsample
		self.output = self._downsample(inpt) * self.scale

	else:            # Upsample
		self.output = self._upsample(inpt) * self.scale

	self.delta = np.zeros(shape=inpt.shape, dtype=float)
```
That calls for the functions `_downsample` and `_upsample` depending on the value of `self.reverse`, respectively `True` or `False`.

And this is the definition of `backward`:

```python
def backward(self, delta):
	'''
	Compute the inverse transformation of the forward function
	on the gradient

	Parameters:
		delta : global error to be backpropagated
	'''

	if self.reverse: # Upsample
		delta[:] = self._upsample(self.delta) * (1. / self.scale)

	else:            # Downsample
		delta[:] = self._downsample(self.delta) * (1. / self.scale)
```
That's just the inverse of the forward.

The real core of the layer though, are the two function `_upsample` and `_downsample`, defined as:

```python
def _upsample (self, inpt):
	batch, w,  h,  c  = inpt.shape     # number of rows/columns
	b,     ws, hs, cs = inpt.strides   # row/column strides

	x = as_strided(inpt, (batch, w, self.stride[0], h, self.stride[1], c), (b, ws, 0, hs, 0, cs)) # view a as larger 4D array
	return x.reshape(batch, w * self.stride[0], h * self.stride[1], c)                            # create new 2D array
```

The up-sample function use the stride functionality of the Numpy array to rearrange and replicate the value of each pixel in a mask of size `strides × strides`.

And here's the `_downsample` function:

```python
def _downsample (self, inpt):
	# This function works only if the dimensions are perfectly divisible by strides
	# TODO: add padding (?)
	batch, w, h, c = inpt.shape
	scale_w = w // self.stride[0]
	scale_h = h // self.stride[1]

	return inpt.reshape(batch, scale_w, self.stride[0], scale_h, self.stride[1], c).mean(axis=(2, 4))
```

The down-sampling algorithm is obtained reshaping the input array according to two scale factors (`strides` in the code) along the two dimensions and computing the mean along these axes.

Unfortunately, for now it works only if `h % stride` and `w % stride` are zero.
