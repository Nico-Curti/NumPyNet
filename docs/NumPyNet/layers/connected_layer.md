#### Connected Layer

The connected layer, or dense layer, is the single matrix operation performed by a layer of a Fully Connected Neural Network.
Given a batch of images arranged in a matrix *X* of shape (batch, image_size) and given the set of trainable weights *W* of shape (image_size, output_size) and the bias vector *b* of shape (output_size), the connected layer computes the linear operation:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=z&space;=&space;XW&space;&plus;&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z&space;=&space;XW&space;&plus;&space;b" title="z = XW + b" /></a>
</p>

To develop non-linearity and improve the learning capabilities of the layer, `z` is "activated" by an activation function, so that the final output of the layer will be:

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;f(z)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;f(z)" title="y = f(z)" /></a>
</p>

Tha backward function computes the updates for weights and bias and the error to be backpropagated:





Here there's an example on how to use the single connected layer forward, backward and update functions:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.connected_layer import Connected_layer # class import
from NumPyNet import activations

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3) # batch != 1 in this case
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import some images from file

inputs  = w * h * c
outputs = 10        # arbitrary

weights    = np.random.uniform(low=0., high=1., size=(inputs, outputs))
bias       = np.random.uniform(low=0., high=1., size=(outputs,))
activ_func = activations.Relu() # it can also be:
                                #     activations.Relu  (class Relu)
                                #     "Relu" (a string, case insensitive)

# Layer initialization, with parameters scales and bias
layer = Connected_layer(input_shape=input.shape, outputs=outputs,
                        weights=weights, bias=bias, activation=activ_func)

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

To have a look in details on what's happening inside every function, this is the forward:
```python
def forward(self, inpt, copy=False):
  '''
  Forward function of the connected layer. It computes the matrix product
    between inpt and weights, add bias and activate the result with the
    chosen activation function.
  Parameters:
    inpt : numpy array with shape (batch, w, h, c) input batch of images of the layer
    copy : boolean, states if the activation function have to return a copy of the
          input or not.
  '''

  inpt = inpt.reshape(-1, self.inputs)                  # shape (batch, w*h*c)

  #z = (inpt @ self.weights) + self.bias                # shape (batch, outputs)
  z = np.einsum('ij, jk -> ik', inpt, self.weights, optimize=True) + self.bias
  #z = np.dot(inpt, self.weights) + self.bias

  self.output = self.activation(z, copy=copy)           # shape (batch, outputs), activated
  self.delta = np.zeros(shape=self.out_shape, dtype=float)
```




Backward function

```python
def backward(self, inpt, delta=None, copy=False):
  '''
  Backward function of the connected layer, updates the global delta of the
    network to be Backpropagated, he weights upadtes and the biases updates
  Parameters:
    inpt  : original input of the layer
    delta : global delta, to be backpropagated.
    copy  : boolean, states if the activation function have to return a copy of the
          input or not.
  '''

  # reshape to (batch , w * h * c)
  inpt = inpt.reshape(self._out_shape[0], -1)

  self.delta *= self.gradient(self.output, copy=copy)

  self.bias_update += self.delta.sum(axis=0)   # shape : (outputs,)

  # self.weights_update += inpt.transpose() @ self.delta') # shape : (w * h * c, outputs)
  self.weights_update += np.dot(inpt.transpose(), self.delta)

  if delta is not None:
    delta_shaped = delta.reshape(self._out_shape[0], -1)  # it's a reshaped VIEW

    # shapes : (batch , w * h * c) = (batch , w * h * c) + (batch, outputs) @ (outputs, w * h * c)

    # delta_shaped[:] += self.delta @ self.weights.transpose()')  # I can modify delta using its view
    delta_shaped[:] += np.dot(self.delta, self.weights.transpose())

```
