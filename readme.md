# Optimizing a Neural Network Parameters with PSO

To run the default experiments run main.py

## Create a network.

Create a neural net with hidden layers.

```
import ANN.NeuralNet

# Create a neural network with 2 layers with 4 nodes and 1 output.
network = NeuralNet()
network.add_layer(input_count=1, node_count=4, activations=[0,2,3,4])
network.add_layer(input_count=4, node_count=1, activations=[2])
```

Create a NeuralNet with NetworkGenerator
```
from ANN.networkGenerator import NetworkGenerator

# Start the generator and add desired layer size
net_generator = NetworkGenerator()
net_generator.add_layer(input_count=1, node_count=4, activations=[0,2,3,4])
net_generator.add_layer(input_count=4, node_count=1, activations=[2])

# Generate a neural net with those layer parameters
network = net_generator.generate_network()
```
## Run the PSO
Create a PSO algorithm to improve your network parameters.

```
pso = PSO(
        net_generator = self.network,
        swarmsize = self.pso_params["swarmsize"],
        alpha = self.pso_params["alpha"],
        beta = self.pso_params["beta"],
        gamma = self.pso_params["gamma"],
        delta = self.pso_params["delta"],
        jumpsize = self.pso_params["jumpsize"],
        act_bound = self.pso_params["act_bound"],
        weight_bound = self.pso_params["weight_bound"],
        num_informants = self.pso_params["num_informants"],
        max_runs = self.pso_params["max_runs"],
        ideal=self.ideal,
        inputs=self.inputs)

# Run the pso algorithm
pso.run_algo()
```

# Create an Experiment

Create an experiment to run the pso with different parameters.

```
# Set the paramaters for the pso
params_pso = {
    "swarmsize": 50,
    "alpha": 0.5,
    "beta": 0.5,
    "gamma": 0.5,
    "delta": 0.5,
    "jumpsize": 0.5,
    "act_bound": 5,
    "weight_bound": 0.5,
    "num_informants": 50,
    "max_runs": 100
}

# Set the parameters for Neural Net
net_layers = {
    "layer1": {
        "input_count":1,
        "node_count":8,
        "activations": []
    },
    "layer2": {
        "input_count":8,
        "node_count": 1,
        "activations:":[]
    }
}

# Create the experiment and run it.
experiment = Experiment(params_pso, net_layers, path="1in_linear.txt")
experiment.run()
```