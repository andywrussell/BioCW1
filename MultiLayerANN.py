import math
# Hlello
class Synapse:
    def __init__(self, in_val, weight):
        self.in_val = in_val
        self.weight = weight

class Node :
    def __init__(self, bias, layerIndex, nodeIndex):#, activationfunction) :
       ## self.synapses = synapses
        self.bias = bias
        self.layerIndex = layerIndex
        self.nodeIndex = nodeIndex
     #   self.activationfunction = activationfunction

    ##Do the summing
    def TakeInput(self, inputs):
        self.in_synapse = []
        for in_val in inputs:
            self.in_synapse.append(Synapse(in_val, 1))

        self.Activate()

    def NodeSum(self):
        self.linearcomb = 1 * self.bias
         ##No need to do the transpose just now as we are using 1 dimensional vectors
        for synapse in self.in_synapse:
           self.linearcomb += synapse.weight * synapse.in_val 
    
    def Activate(self):
        self.NodeSum()
        self.output = 1 / (1 - math.exp(-self.linearcomb)) # self.activationfunction(self.linearcomb)

class Layer:
    def __init__ (self, index, nodeCount):
        self.Nodes = [Node(1, index, x) for x in range(nodeCount)]
        self.index = index

    def SetInput(self, inputs):
        for node in self.Nodes:
            node.TakeInput(inputs)


class GloriousANN:
    def __init__ (self, layerCount, nodesPerLayer):
        self.layers = [Layer(x, nodesPerLayer) for x in range(nodesPerLayer)]

    def SetInput(self, inputs):
        ##set input on first layer
        self.layers[0].SetInput(inputs)
        i = 1
        while (i < len(self.layers)) :
            
        



#ANN = GloriousANN(3, 3)
  




        