from ANN.networkGenerator import NetworkGenerator
from PSO.pso import PSO
from utils.helpers import MSE, read_data
import os
import pandas as pd

"""
Class Experiment takes some parameters and runs a pso.
"""
class Experiment:
    def __init__(self, pso_params, net_params, path, debugMode=True, sampleMode=False, net_generator=NetworkGenerator, PSO=PSO):
        self.pso_params = pso_params
        self.net_params = net_params
        self.path = path
        self.debugMode = debugMode
        self.sampleMode = sampleMode

        self.pso = None
        self.network = None

        self.load_data(self.path)
        self.build_network(net_generator)
        self.build_pso(PSO)



    def build_network(self, net_generator):
        self.network = net_generator()
        
        for key, val in self.net_params.items():
            self.network.add_layer(
                input_count=val["input_count"],
                node_count=val["node_count"], activations=[])
    
    def build_pso(self, PSO):
        assert len(self.network.layers) != 0, "Network must be initialized before initializing the pso"

        self.pso = PSO(
            net_generator = self.network,
            swarmsize = self.pso_params["swarmsize"],
            alpha = self.pso_params["alpha"],
            beta = self.pso_params["beta"],
            gamma = self.pso_params["gamma"],
            delta = self.pso_params["delta"],
            jumpsize = self.pso_params["jumpsize"],
            act_bound = self.pso_params["act_bound"],
            weight_bound = self.pso_params["weight_bound"],
            bound_strat = self.pso_params["bound_strat"],
            num_informants = self.pso_params["num_informants"],
            max_runs = self.pso_params["max_runs"],
            vel_range = self.pso_params["vel_range"],
            informant_strat = self.pso_params["informants_strat"],
            ideal=self.ideal,
            inputs=self.inputs)

    def load_data(self, path):
        current_dir = os.getcwd() + '/'
        self.inputs, self.ideal = read_data(current_dir, path, self.sampleMode)

        if (self.debugMode):
            self.inputs = self.inputs.head(10)
            self.ideal = self.ideal.head(10)
        
    def print_results(self, outputs=True, net=False):
        # Print outputs
        if (outputs):
            results = self.pso.best.outputs
            df = self.ideal.copy()
            df['results'] = results
            print(df.head(10))
        # Print the net
        if (net):
            self.pso.best.network.print_net()

    def run(self):
        self.pso.run_algo()
        self.print_results()