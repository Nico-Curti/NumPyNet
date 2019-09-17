### Activation Layer

Activation functions (or transfer functions) are linear or non linear equations which process the output of a Neural Network's neuron and bound it into a limited range of values (usually [-1,1] or [0,1]).

In a fully connected neural network, the output of a simple neuron is the dot product between weight and input vectors, ranging from to -&infin; to +&infin;  but most importantly the output is the result of a linear function.
Linear functions are very simple to be dealt with, but they are limited in their complexity and thus in their learning power.
Neural Networks without activation functions are just simple linear regression model.
The introduction of non-linearity allow them to model a wide range of functions and to learn more complex relations in the pattern data.
From a biological point of view, the activation function models the on/off state of a neuron in the output decision process.

Many activation functions were proposed during the years and each one has its characteristics but not an appropriate field of application.
The better one to use in a particular situation is still an open question.
Each one has its pro and cons, so each Neural Network libraries implements a wide range of activations and leaves the user to perform his own tests.

We stored the whole list of activation functions available in [activations.py](https://github.com/Nico-Curti/NumPyNet/blob/master/NumPyNet/activations.py), with their own formulations and derivatives.
An important feature of any activation function, in fact, is that it should be differentiable, since the main procedure of model optimization (Learning) implies the backpropagation of the error gradients.

The images below show some examples about the effect of the forward and backward pass of the activation layer on the same input picture:

![](https://github.com/Nico-Curti/NumPyNet/blob/master/docs/NumPyNet/images/activation_relu.png)
![](https://github.com/Nico-Curti/NumPyNet/blob/master/docs/NumPyNet/images/activation_logistic.png )
![](https://github.com/Nico-Curti/NumPyNet/blob/master/docs/NumPyNet/images/activation_elu.png)
*Fig. 1: examples of activation functions being applied to the same input image. From up to down: Relu, Logistic and Elu, for both forward pass and backward pass*

The code used to generate those images can be found [in this repository](https://github.com/Nico-Curti/NumPyNet/blob/master/NumPyNet/layers/activation_layer.py "activation_layer.py"), after the activation layer class definition.

Below is shown an example on how to use the single layer to perform its `forward` and `backward` function:

```python
# first the essential import for the library.
# after the installation:
from NumPyNet.layers.activation_layer import Activation_layer # class import
from NumPyNet import activations                              # here are contained all the activation funtions definitions

import numpy as np # the library is entirely based on numpy

# define a batch of images (even a single image is ok, but is important that it has all
# the four dimensions) in the format (batch, width, height, channels)

batch, w, h, c = (5, 100, 100, 3)
input = np.random.uniform(low=0., high=1., size=(batch, w, h, c)) # you can also import an image from file

# Activation function definition
Activ_func = activations.Relu() # it can also be:
                                #    activations.Relu (the class Relu, taken from activations.py)
                                #    'Relu' (a string)

# Layer initialization
layer = Activation_layer(activation=Activ_func)

# Forward pass
layer.forward(inpt=input, copy=False)
out_img = layer.output    # the output in this case will be of shape=(batch, w, h, c), so a batch of images


# Backward pass
delta       = np.random.uniform(low=0., high=1., size=input.shape)     # definition of network delta, to be backpropagated
layer.delta = np.random.uniform(low=0., high=1., size=out_img.shape) # layer delta, ideally coming from the next layer
layer.backward(delta, copy=False)

# now net_delta is modified and ready to be passed to the previous layer.delta
```

To have an idea on what the forward and backward function actually do, take a look at the code below:

###### Forward function:

```python
def forward(self, inpt, copy=True):
  '''
  Forward of the activation layer, apply the selected activation function to
  the input

  Parameters:
    inpt: the input to be activated
    copy: default value is True. If True make a copy of the input before
          applying the activation
  '''
  self._out_shape = inpt.shape
  self.output     = self.activation(inpt, copy=copy)
  self.delta      = np.zeros(shape=self.out_shape, dtype=float)

```
The code is very straight-forward:
1. store the variable
2. apply the selected activation function to the input
3. initialize layer.delta to all zero.

###### Backward function:

```python
def backward(self, delta, copy=False):
  '''
  Compute the backward of the activation layer

  Parameter:
    delta : global error to be backpropagated
  '''

  self.delta *= self.gradient(self.output, copy=copy)
  delta[:] = self.delta
```
Here instead :
1. multiply `layer.delta` for the derivative of the activation function (computed on the **activated** output)image
2. modify delta with the current value of `layer.delta`.
