# Cost layer

In order to understand and quantify how well our model makes prediction on data, we define a *cost function*, which is a measure of the error committed by the network during training, and, for this purpose, the *cost layer* is often the last one in a CNN.
An easy example of cost function is the *mean squared error* (or L2 norm):

![](https://latex.codecogs.com/gif.latex?C&space;=\frac{1}{N}&space;\sum_{i=0}^{N}&space;(y_i&space;-&space;t_i)^2)

where `y` is the predicted vector, while `t` is the array of true labels.
It's clear that an higher difference between prediction and truth, produces an higher cost.
The minimization of this function is the objective of the backpropagation algorithm.
Of course, a labeled dataset is needed to compute the cost (and, as a consequence, to train the Network).

**TODO**: explain which cost functions we implemented
The definition of the differents cost function is inside the `Cost_layer` class in [this repository](https://github.com/Nico-Curti/NumPyNet/blob/master/NumPyNet/layers/cost_layer.py)

Here an example on how to use the cost layer as a single layer:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.cost_layer import Cost_layer      # class import
from NumPyNet.layers.cost_layer import cost_type as cs # cost_type enum class import

import numpy as np # the library is entirely based on numpy

batch, w, h, c = (5, 100, 100, 3)
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # usually a vector or an image

cost = cs.mse # Mean squared error function.

# Layer initialization, with parameters scales and bias
layer = Cost_layer(input_shape=input.shape,
                   cost_type=,
                   scale=1.,
                   ratio=0.,
                   noobject_scale=1.,
                   threshold=0.,
                   smoothing=0.)

# Forward pass
layer.forward(inpt=input, copy=False)
out_img = layer.output    # the output in this case will be the same shape of the input

# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)   # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified and ready to be passed to the previous layer.delta
```

To have a look more in details on what's happening, the definitions of `forward` and `backward` functions:

```python
def forward(self, inpt, truth=None):
  '''
  Forward function for the cost layer. Using the chosen
  cost function, computes output, delta and cost.
  Parameters:
    inpt: the output of the previous layer.
    truth: truth values, it should have the same
      dimension as inpt.
  '''
  self._out_shape = inpt.shape

  if truth is not None:

    if self.smoothing: self._smoothing(truth)                              # smooth is applied on truth

    if   self.cost_type == cost_type.smooth:    self._smooth_l1(inpt, truth)  # smooth_l1 if smooth not zero
    elif self.cost_type == cost_type.mae:       self._l1(inpt, truth)         # call for l1 if mae is cost
    elif self.cost_type == cost_type.wgan:      self._wgan(inpt, truth)       # call for wgan
    elif self.cost_type == cost_type.hellinger: self._hellinger(inpt, truth)  # call for hellinger distance
    elif self.cost_type == cost_type.hinge:     self._hinge(inpt, truth)  # call for hellinger distance
    elif self.cost_type == cost_type.logcosh:   self._logcosh(inpt, truth)  # call for hellinger distance
    else:                                       self._l2(inpt, truth)         # call for l2 if mse or nothing

    if self.cost_type == cost_type.seg and self.noobject_scale != 1.:      # seg if noobject_scale is not 1.
      self._seg(truth)

    if self.cost_type == cost_type.masked:                                 # l2 Masked truth values if selected
      self._masked(inpt, truth)

    if self.ratio:                                                         #
      self._ratio(truth)

    if self.threshold:                                                     #
      self._threshold()


    norm = 1. / self.delta.size                                            # normalization of delta!
    self.delta *= norm

    self.cost = np.mean(self.output)                                       # compute the cost
```

The code proceeds as follow, if the truth array is given:

  * The first part is a "switch" that apply the selected cost function. In every cost function output and delta are updated (Note that `layer.output` and `layer.delta` have always the same dimensions as the input). In the case of a *mean squared error* function, out<sub>i</sub> will be (x<sub>i</sub> - t<sub>i</sub>)<sup>2</sup>
  * In the second part a series of method is applied to input, output or the truth array, if the respective variable has the correct value.
  * In the last part, `layer.delta` is normalized and `layer.cost` is computed as the mean of the output.

And here's the backward function:

```python
def backward(self, delta):
  '''
  Backward function of the cost_layer, it updates the delta
  variable to be backpropagated. self.delta is updated inside the cost function.
  Parameters:
    delta: array, error of the network, to be backpropagated
  '''
  delta[:] += self.scale * self.delta
```

That's just an update of `delta` with a scaled `layer.delta`
