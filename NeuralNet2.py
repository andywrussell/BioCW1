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

    def activation(self, in_val) :
        return 1 / (1 - math.exp(-in_val))

    def fire_layer(self, in_vals):
        outputs = in_vals.dot(self.weights)
        
        # Apply the activation for the output of each neuron
        for i, output in enumerate(outputs):
            activation = activation_dict[self.activations[i]]
            outputs[i] = activation(output)
        self.outputs = outputs
        

        

class NeuralNet:
    def __init__(self, layers, inputs) :
        self.layers = layers
        self.inputs = inputs


    def fire_net(self):
        layer_input = self.inputs

        for layer in self.layers:
            input_with_bias = np.append(layer_input, [1])
            layer.fire_layer(input_with_bias)
            layer_input = layer.outputs

        self.output = layer_input
    


        