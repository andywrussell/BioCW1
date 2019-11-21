from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from utils.helpers import MSE 
"""
This class saves the parameters we want to use in the network.
It can generate a Neural Net that we assign to each particle.
"""
class NetworkGenerator:
    def __init__(self, error_function=MSE, neural_net=NeuralNet):
        self.layers = []
        self.NeuralNet = NeuralNet
    
    def add_layer(self, input_count, node_count, activations=[]):
        layer = Layer(input_count , node_count, activations)
        layer.build_layer()
        
        self.layers.append(layer)

    def generate_network(self):
        network = NeuralNet(error_function = MSE)
        network.layers = self.layers
        return network