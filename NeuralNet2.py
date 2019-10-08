import numpy as np
import math

        
class Layer:
    def __init__(self, inputs, nodes) :
        self.inputs = inputs
        self.nodes = nodes

    def BuildLayer(self) :
        inputsWithBias = self.inputs + 1
        self.weights = np.random.rand(inputsWithBias, self.nodes)

    def Activation(self, in_val) :
        return 1 / (1 - math.exp(-in_val))

    def Run(self, in_vals):
        self.outputs = in_vals.dot(self.weights)

        

class NeuralNet:
    def __init__(self, layers, inputs) :
        self.layers = layers
        self.inputs = np.append(inputs, [1])


    def forwardPass(self):
        self.layers[0].Run(self.inputs)

    


        