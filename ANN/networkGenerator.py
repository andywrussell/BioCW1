from ANN.neuralNet import NeuralNet
from ANN.layer import Layer
from utils.helpers import MSE, SUM

"""
This class saves the parameters we want to use in the network.
It can generate a Neural Net that we assign to each particle.
"""
class NetworkGenerator:
    def __init__(self, error_function=MSE, neural_net=NeuralNet):
        """
        Params
        ======
        layers: layers we want the network to have
        NeuralNet: The net object we will return.
        """
        self.layers = []
        self.NeuralNet = NeuralNet
    
    def add_layer(self, input_count, node_count, activations=[]):
        """
        Adds a layer to the network generator.
        """
        layer = Layer(input_count , node_count, activations)
        layer.build_layer()
        
        self.layers.append(layer)

    def generate_network(self):
        """
        Returns a NeuranNetwork instance with the layers specified in this object.
        """
        network = NeuralNet(error_function = MSE)
        mylayers = []
        for layer in self.layers:
            layer = Layer(layer.input_count , layer.node_count, layer.activations)
            layer.build_layer()
            mylayers.append(layer)
            
        network.layers = mylayers
        return network