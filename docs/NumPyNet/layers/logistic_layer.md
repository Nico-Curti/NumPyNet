# Logistic Layer

The logistic layer is a particular implementation of what has already been describe for the [Cost Layer](./cost_layer.md).
It perfmors a logitic tranformation of the output as:

![](https://latex.codecogs.com/gif.latex?y&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}})

and then, if its `forward` function recevives `truth` values, it computes the binary cross entropy loss as:

![](https://latex.codecogs.com/gif.latex?L&space;=&space;-t&space;\cdot&space;log(y)&space;-&space;(1&space;-&space;t)&space;\cdot&space;log(1&space;-&space;y))

and the `cost`:

![](https://latex.codecogs.com/gif.latex?C&space;=&space;\sum_{i=0}^{N}&space;L_i)

where `N` is the toal number of features in the input `x`.

This is an example code on how to use the single `Logistic_layer` object:

```python
# first, the essential import for the layer.
from NumPyNet.layers.logistic_layer import Logistic_layer

import numpy as np # the library is entirely based on numpy

batch, w, h, c = (5, 100, 100, 3)
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # ideally is the output of a Model
truth = np.random.choice(a=[0., 1.], size=input.shape, p=None) # Binary truth, p=None assume a uniform ditro between "a"

layer = Logistic_layer()

# forward
layer.forward(inpt=input, truth=truth )
output = layer.output # this is the tranformed input
cost   = layer.cost		# real number, a measure of how well the model perfomed
loss   = layer.loss		#	loss of the Model, every element is the distance of output form the truth.

# backward
delta 			= np.zeros(shape=input.shape)
layer.backward(delta=delta) # layer.delta is already initialized in forward

# now delta is updated and ready to be passed backward
```

To have a look more in details on what's happening, the definitions of `forward` and `backward`:

```python
def forward(self, inpt, truth=None) :
	'''
	Forward function of the logistic layer, now the output should be consistent with darknet

	Parameters:
		inpt : output of the network with shape (batch, w, h, c)
		truth : arrat of same shape as input (without the batch dimension),
			if given, the function computes the binary cross entropy
	'''

	self._out_shape = inpt.shape
	# inpt = np.log(inpt/(1-inpt))
	self.output = 1. / (1. + np.exp(-inpt)) # as for darknet
	# self.output = inpt

	if truth is not None:
		out = np.clip(self.output, 1e-8, 1. - 1e-8)
		self.loss = -truth * np.log(out) - (1. - truth) * np.log(1. - out)
		out_upd = out * (1. - out)
		out_upd[out_upd <= 1e-8] = 1e-8
		self.delta = (truth - out) * out_upd
		# self.cost = np.mean(self.loss)
		self.cost = np.sum(self.loss) # as for darknet
	else :
		self.delta = np.zeros(shape=self._out_shape, dtype=float)
```

The code proceeds as follow:

  * `self.output` is computed as the element-wise sigmoid tranformation of the input.
  * If `truth` is given (same shape as `inpt`), then the function clip `self.output` in the range [&epsilon;, 1-&epsilon;], this is due to the singularity of the logarithm.
  * `self.loss` is computed as described above.
  * `self.delta` is updated as:

![](https://latex.codecogs.com/gif.latex?\delta&space;=&space;(t&space;-&space;y)&space;\cdot&space;y(1&space;-&space;y))

  * and `self.cost` is the sum of all `self.loss` elements

And this is the `backward` definiton:

```python
def backward(self, delta=None):
	'''
	Backward function of the Logistic Layer

	Parameters:
		delta : array same shape as the input.
	'''
	if delta is not None:
		delta[:] += self.delta # as for darknet, probably an approx
```

That is a simple update of `delta` with values of `self.delta` computed in `forward`.
