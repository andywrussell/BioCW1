import numpy as np
import math

class Node:
    def __init__(self, weights):
        self.weights = weights

    def fire_node(self, inputs):
        linearcomb = inputs.dot(self.weights)
        ##we should replace this with a changeable activation function
        self.output = 1 / (1 - math.exp(-linearcomb)) 

class Layer:
    def __init__(self, input_count, node_count) :
        self.input_count = input_count
        self.node_count = node_count
        self.build_layer()

    def build_layer(self):
        self.nodes = []
        for i in range(self.node_count):
            inputs_with_bias = self.input_count + 1
            node_weights = np.random.rand(inputs_with_bias)
            self.nodes.append(Node(node_weights))

    def fire_layer(self, inputs):
        self.outputs = []
        for node in self.nodes:
            node.fire_node(inputs)
            self.outputs.append(node.output)

class NeuralNet:
    def __init__(self, layers, inputs) :
        self.layers = layers
        self.inputs = inputs

    def fire_net(self):
        layerinput = self.inputs
        for layer in self.layers:
            input_with_bias = np.append(layerinput, [1])
            layer.fire_layer(input_with_bias)
            layerinput = layer.outputs

        self.output = layerinput