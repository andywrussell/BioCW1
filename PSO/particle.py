class Particle:
    def __init__(self, network, position, velocity, ideal, inputs):
        """
        Params
        ======
        * network: ANN that will be trained by this particle
        * position: starting position of the particle (flattend ANN)
        * velocity: initial velocity
        * ideal: desired output of network
        * inputs: input of network
        """
        
        self.position_list = [position] #all of the positions so far
        self.velocity_list = [velocity] #all of the velocities used so far
        self.network = network  
        self.position = position
        self.velocity = velocity
        self.inputs = inputs
        self.outputs = []   #current output of ann
        self.ideal = ideal
        self.best_list = [position] #list of all the best positions
        self.best = position #current best position
        self.best_fitness = None #current error for best position
        self.asses_fitness()


    def update_position(self, new_pos):
        """
        Update the particle velocity
        Add the new position to the list and update the network
        """
        self.position_list.append(new_pos)        
        self.position = new_pos
        self.network.flatten_net()
        self.network.net_as_vector = new_pos
        self.network.unflatten_net()
        
    def update_velocity(self, new_vel):
        """
        Update the particle velocity
        """
        self.velocity_list.append(new_vel)
        self.velocity = new_vel

    def asses_fitness(self):
        self.fitness, self.outputs = self.network.get_fitness(self.inputs, self.ideal)
        
        #otherwise get distance from ideal and see if it is better than current best
        if (self.best_fitness == None or self.fitness < self.best_fitness):
            self.best = self.position
            self.best_list.append(self.position)
            self.best_fitness = self.fitness            
    
                
    def informants_best(self):
        """
        Get the previous best position from all informants including this particle
        """
        inf_best = self.informants[0]        
        for inf in self.informants:
            if (inf.best_fitness < inf_best.best_fitness):
                inf_best = inf

        if inf_best.best_fitness < self.best_fitness:
            return inf_best.best_list[-2]
        else:
            return self.best_list[-2]

