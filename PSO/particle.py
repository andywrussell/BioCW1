class Particle:
    def __init__(self, network, position, velocity, ideal, inputs):
        #the index of a position or velocity in the list corresponds to a timestamp
        self.position_list = [position]
        self.velocity_list = [velocity]
        self.network = network
        self.position = position
        self.velocity = velocity
        self.inputs = inputs
        self.outputs = []
        self.ideal = ideal
        self.best_list = [position]
        self.best = position
        self.best_fitness = None
        self.asses_fitness()


    def update_position(self, new_pos):
        self.position_list.append(new_pos)        
        self.position = new_pos
        self.network.flatten_net()
        self.network.net_as_vector = new_pos
        self.network.unflatten_net()
        
    def update_velocity(self, new_vel):
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
        inf_best = self.informants[0]        
        for inf in self.informants:
            if (inf.best_fitness < inf_best.best_fitness):
                inf_best = inf

        if inf_best.best_fitness < self.best_fitness:
            return inf_best.best_list[-2]
        else:
            return self.best_list[-2]

