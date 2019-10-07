class Synapse:
    def __init__(self, in_val, weight, index):
        self.in_val = in_val
        self.weight = weight
        self.index = index

class Node :
    def __init__(self, synapses, bias, activationfunction) :
        self.synapses = synapses
        self.bias = bias
        self.activationfunction = activationfunction

    ##Do the summing
    def NodeSum(self):
        self.linearcomb = 1 * self.bias
        ##No need to do the transpose just now as we are using 1 dimensional vectors
        for synapse in self.synapses:
           self.linearcomb += synapse.weight * synapse.in_val 
    
    def Activate(self):
        self.NodeSum()
        self.output = self.activationfunction(self.linearcomb)

class Layer:
    def __init__ (self, inputs, nodeCount):
        self.inputs = inputs
        




        