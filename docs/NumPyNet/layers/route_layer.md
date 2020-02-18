# Route Layer

In the YOLOv3 model, the Route layer is usefull to bring finer grained features in from earlier in the network.
This mean that its main function is to recover output from previous layer in the network and bring them forward, avoiding all the in-between processes.
Moreover, it is able to recall more than one layer's output, by concatenating them. In this case though, all Route layer's input must have the same width and height.
It's role in a CNN is similar to what has alredy been described for the [Shortcut layer](./shortcut_layer.md).

In the YOLOv3 applications, it's always used to concatenate outputs by channels: let out1 = (batch, w, h, c1) and out2 = (batch, w, h, c2) be the two inputs of the Route layer, then the final output will be a tensor of shape (bacth, w, h, c1 + c2), as described [here](https://github.com/AlexeyAB/darknet/issues/487).
On the other hand, the popular Machine Learning library Caffe, let the user chose, by the [Concat Layer](https://caffe.berkeleyvision.org/tutorial/layers/concat.html), if the concatenation must be performed channels or batch dimension, in a similar way as described above.

Our implementation is similar to Caffe, even though the applications will have more resamblance with YOLO models.

An example on how to instantiate a Route layer and use it is shown in the code below:

```python

from NumPyNet.layers.route_layer import Route_layer
import numpy as np # the library is entirely based on numpy

from NumPyNet.network import Network

model = Network() # imagine that model is a CNN with 10 layers

 # [...] istantiation of model, via model.add or cgf file

# layer init
layer = Route_layer(inpt_layers=(3,6), by_channels=True)

# forward
layer.forward(network=model) # Assuming layer 3 and 6 have the same dimensios batch, width and height, the output will be (batch, w, h, c1 + c2)
output = layer.output

# backward
delta = np.random.uniform(low=0., high=1., size=layer.output.shape) # delta coming from next layers, ideally
layer.backward(delta=delta, network=model)

# now backward updates only the self.delta of layer 3 and 6 of model, there's no layer.delta, since is not needed
```
Of course, there are smarter ways of using this layer, as demostrated by YOLOv3.
The parameter `by_channels` determines `self.axis` (if True is 3, else is 0), to perfmorm the correct concatenation.

In particular, those are the definitions of the `forward` and `backward` functions in `NumPyNet`:

```python
def forward(self, network):
	'''
	Concatenate along chosen axis the outputs of selected network layers
	In main CNN applications, like YOLOv3, the concatenation happens channels wise

	Parameters:
		network : Network object type.
	'''

	self.output = np.concatenate([network[layer_idx] for layer_idx in self.input_layers], axis=self.axis)
```

Where `self.input_layers` is the list of indexes at which the chosen layers are located in the network (starting at 1, since 0 is always an [Input Layer](./input_layer.md)). As you can see, is a simple concatenation by the correct axis.

And this is the definition of `backward`:

```python
def backward(self, delta, network):
	'''
	Sum self.delta to the correct layer delta on the network

	Parameters:
		delta  : 4-d numpy array, network delta to be backpropagated
		network: Network object type.
	'''

	if self.axis == 3:            # this works for concatenation by channels axis
		channels_sum = 0
		for idx in self.input_layers:
			channels = network[idx].out_shape[3]
			network[idx].delta += delta[:,:,:, channels_sum : channels_sum + channels]
			channels_sum += channels

	elif self.axis == 0:          # this works for concatenation by batch axis
		batch_sum = 0
		for idx in self.self.input_layers:
			batches = network[idx].out_shape[0]
			network[idx].delta += delta[batch_sum : batch_sum + batches,:,:,:]
			batch_sum += batches
```

in this case, `self.delta` of the correspoding layer is updated taking into consideration the dimensions: if the first layer has 3 channels, the only the first 3 channels of `delta` are passed to it.
