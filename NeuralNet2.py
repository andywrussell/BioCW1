import numpy as np
import math
from ActivationFunctions import activation_dict


class Layer:
    def __init__(self, input_count, node_count, activations):
        """
        Params
        ======
        * input_count = number of inputs
        * node_count = number of nodes in the layer
        * activations = activations for each node
        """

        self.input_count = input_count
        self.node_count = node_count
        self.activations = np.asarray(activations)

    def build_layer(self) :
        inputsWithBias = self.input_count + 1
        self.weights = np.random.rand(inputsWithBias, self.node_count)
        self.weights_and_activations = (self.weights, self.activations)

    def fire_layer(self, in_vals):
        outputs = in_vals.dot(self.weights)
        
        # Apply the activation for the output of each neuron
        for i, output in enumerate(outputs):
            activation = activation_dict[self.activations[i]]
            outputs[i] = activation(output)
        self.outputs = outputs
            

class NeuralNet:
    def __init__(self, layers, inputs) :
        """
        Params
        ======
        * layers: an array that contains layer objects
        * inputs: a single input from our dataset (1 row of our inputs)
        * net_as_vector: the vector version of our net (aka particle)
        * net_shape: Array of arrays containing the shape of the layer and its activations.
            each array in net_shape has this form [(shape of weights), (shape of activations)]
        """
        self.layers = layers
        self.inputs = inputs
        self.net_as_vector = []
        self.net_shape = []


    def fire_net(self):
        layer_input = self.inputs

        for layer in self.layers:
            input_with_bias = np.append(layer_input, [1])
            layer.fire_layer(input_with_bias)
            layer_input = layer.outputs

        self.output = layer_input

    def flatten_array(self, layer):
        numpy_flatten = layer.flatten()
        list_flatten  = numpy_flatten.tolist()
        return list_flatten
    
    def flatten_net(self):
        """
        Params
        ======
        * weights_shape: the original shape of the weights. Used later to un-flatten the net
        * activations_shape: the original shape of the activations. Used to recover the matrix
        * layer_shape: An array of 2 tuples. First tuple is the weights_shape, and second tuple is activations_shape
        """
        for layer in self.layers:

            weights_shape = layer.weights.shape
            activations_shape = layer.activations.shape
            
            flatten_weights = self.flatten_array(layer.weights)
            flatten_activations = self.flatten_array(layer.activations)
            layer_shape = [weights_shape, activations_shape]

            self.net_as_vector = self.net_as_vector + flatten_weights
            self.net_as_vector = self.net_as_vector + flatten_activations
            self.net_shape.append(layer_shape)







        