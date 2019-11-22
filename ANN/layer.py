from utils.activationFunctions import activation_dict
import numpy as np
import random

class Layer:
    def __init__(self, input_count, node_count, activations=[]):
        """
        Params
        ======
        * input_count = number of inputs
        * node_count = number of nodes in the layer
        * activations = activations for each node
        """

        self.input_count = input_count
        self.node_count = node_count

        if (len(activations) == 0):
            rand_activations = [random.randint(0, self.node_count) for i in range(self.node_count)]
            self.activations = np.asarray(rand_activations)
        else:
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