import numpy as np
import math
from ANN.layer import Layer
from ActivationFunctions import activation_index
from prettytable import PrettyTable

class NeuralNet:

    def __init__(self, error_function, layers=[]) :
        """
        Params
        ======
        * layers: an array that contains layer objects
        * inputs: a single input from our dataset (1 row of our inputs)
        * net_as_vector: the vector version of our net (aka particle)
        * net_shape: Array of arrays containing the shape of the layer and its activations.
            each array in net_shape has this form [(shape of weights), (shape of activations)]
        """
        self.layers = []
        self.net_as_vector = []
        self.net_shape = []
        self.error_function = error_function

    def add_layer(self, input_count, node_count, activations=[]):
        layer = Layer(input_count , node_count, activations)
        layer.build_layer()
        
        self.layers.append(layer)

    def fire_net(self, layer_input):
        for layer in self.layers:
            input_with_bias = np.append(layer_input, [1])
            layer.fire_layer(input_with_bias)
            layer_input = layer.outputs

        self.output = layer_input

    def get_fitness(self, inputs, outputs):
        errors = np.zeros(inputs.shape[0])
        predictions = np.zeros(inputs.shape[0])

        for i , row in inputs.iterrows():
            input = [row[i] for i in range(row.shape[0])]
            self.fire_net(input)

            prediction = self.output
            predictions[i] = prediction

            output = outputs.iloc[i]

            error = self.error_function(output, prediction)
            errors[i] = error

        return (np.mean(errors), predictions)

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
        * layer_shape: A dictionary of 2 tuples. First tuple is the weights_shape, and second tuple is activations_shape
        """
        for layer in self.layers:

            weights_shape = layer.weights.shape
            activations_shape = layer.activations.shape
            
            flatten_weights = self.flatten_array(layer.weights)
            flatten_activations = self.flatten_array(layer.activations)
            layer_shape = {'weights': weights_shape, 'activations': activations_shape}

            self.net_as_vector = self.net_as_vector + flatten_weights
            self.net_as_vector = self.net_as_vector + flatten_activations
            self.net_shape.append(layer_shape)


    def dot_tuple(self, tuple):
        """
        Recieves a tuple and multiplies its elements.
        We use it to get the number of elements in an array from its shape
        Example: matrix with shape (3, 6) has 18 elements.
        """
        product = 1
        for i in tuple:
            product = product * i
        
        return product

    def unflatten_array(self, cur_index, length, shape, isActivation=False):
        """
        Params
        ======
        * cur_index: Starting point to take elements from self.net_as_vector.
        * length: number of elements to take from self.net_as_vector.
        * shape: the shape we want our final array to have.
        * isActivation: if they are activations we might want to treat them differently.
        """
        flat_array = np.array(self.net_as_vector[cur_index : cur_index + length])
        cur_index = cur_index + length
        my_array = flat_array.reshape(shape)
        
        if (isActivation):
            threshold = len(activation_index) - 1
            my_array = my_array.astype(int)
            my_array = np.abs(my_array)
            my_array[my_array > threshold] = threshold

        return (my_array, cur_index)

    def unflatten_net(self):
        cur_index = 0

        for i, layer_shape in enumerate(self.net_shape):
            len_weights = self.dot_tuple(layer_shape['weights'])
            len_activations =  self.dot_tuple(layer_shape['activations'])

            weights_array, cur_index = self.unflatten_array(cur_index, len_weights, layer_shape['weights'])
            activations_array, cur_index = self.unflatten_array(cur_index, len_activations, layer_shape['activations'], isActivation=True)

            # Check that our new array has the same shape as the original.
            assert weights_array.shape == self.layers[i].weights.shape, "new weights array shape is different from original"
            assert activations_array.shape == self.layers[i].activations.shape, "new activation array shape is different from original"

            # Update
            self.layers[i].weights = weights_array
            self.layers[i].activations = activations_array

        # When finished reset our vector.
        self.net_as_vector = []
        self.net_shape = []

    def print_net(self):
        print("Neural Net Structure")
        print("====================")
        for i, layer in enumerate(self.layers):
            print("\nLayer {}".format(i + 1))

            # Create a table
            t_weights = np.transpose(layer.weights)
            header =[" ", "Activation"]
            header = header + ["weight {}".format(j) for j in range(1, layer.weights.shape[0])] + ["weight bias"]
            table = PrettyTable(header)

            for j, weights_row in enumerate(t_weights):
                row_id = "Neuron {}".format(j+1)
                activation = "{} ({})".format(layer.activations[j], activation_index[layer.activations[j]])
                row = [row_id] + [activation] + weights_row.tolist()

                table.add_row(row)
            print(table)








        