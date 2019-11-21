from PSO.particle import Particle
import numpy as np

class PSO:
    def __init__ (self, net_generator, swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, inputs, num_informants, max_runs, boundary) :
        self.net_generator = net_generator # Class that returns a neural net with given layers.
        self.swarmsize = swarmsize #size of the swarm
        self.alpha = alpha #proportion of velocity to be retained
        self.beta = beta #proportion of personal best to be retained
        self.gamma = gamma #proportion of the 'informants' best to be retained
        self.delta = delta #proportion of global best to be retained
        self.jumpsize = jumpsize #jumpsize
        self.num_informants = num_informants #number of randomly selected informants per particle
        self.ideal = ideal
        self.inputs = inputs
        self.max_runs = max_runs
        self.best = None #this is probably not a good idea
        self.boundary = boundary
        self.best_count = 0

    def generate_particles(self):
        self.particles = []
        for i in range(self.swarmsize):
            
            # my_test_input = np.array([1])
            """
            network = NeuralNet(error_function=MSE)
            network.add_layer(input_count=2 , node_count=4, activations=[0,1,2,2])
            network.add_layer(input_count=4 , node_count=1, activations=[0])
            """
            network = self.net_generator.generate_network()
            network.flatten_net()

            particle_pos = network.net_as_vector 
            network.unflatten_net()
            particle_vel = np.array([])

            for j in range(len(particle_pos)):
                particle_vel = np.append(particle_vel, np.random.uniform(-10.0, 10.0))
                
            new_particle = Particle(network, particle_pos, particle_vel, self.ideal, self.inputs)
            self.particles.append(new_particle)


    def assign_informants(self):
        for particle in self.particles:
            particle.informants = []
            for i in range(self.num_informants):
                rand_inf = np.random.choice(self.particles)
                particle.informants.append(rand_inf)
                
    def asses_fitness(self):
        for particle in self.particles:
            particle.asses_fitness()
            if self.best == None:
                self.best = Particle(particle.network, particle.position, particle.velocity, particle.ideal, particle.inputs)
            elif particle.fitness < self.best.fitness:
                self.best.network = particle.network
                self.best.update_position(particle.position)
                self.best.update_velocity(particle.velocity)
                self.best.asses_fitness()
                
    def update_velocity(self): 
        best_pos = self.best.position                        
        
        for particle in self.particles:
            part_best = particle.best
            inf_best = particle.informants_best()
            particle_vel = particle.velocity
            new_vel = []

            b = np.random.uniform(0, self.beta)
            c = np.random.uniform(0, self.gamma)
            d = np.random.uniform(0, self.delta)
                        
            for i in range(len(particle_vel)):                
                b_val = b*(part_best[i] - particle_vel[i]) 
                c_val = c*(inf_best[i] -particle_vel[i])
                d_val = d*(best_pos[i] -particle_vel[i])
                
                new_vel.append(self.alpha*particle_vel[i] + b_val + c_val + d_val)                
                        
            particle.update_velocity(new_vel)
            #print(new_vel)
            
    def update_positions(self):
        for particle in self.particles:
            new_pos = particle.position + (np.dot(self.jumpsize, particle.velocity)) #use dot to multiply jumpsize by velo array
            
            #if it exceeds the boundary reverse velocity
            for i in range(len(new_pos)):
                if (new_pos[i] > self.boundary or new_pos[i] < -self.boundary):
                    particle.velocity[i] = -particle.velocity[i]
                    
            new_pos = particle.position + (np.dot(self.jumpsize, particle.velocity))
                    
            
            particle.update_position(new_pos)
         #   print(particle.position)
            
    def run_algo(self):
        self.generate_particles()
        self.assign_informants()
        runs = 0
        while (self.best == None or (self.best.fitness > 0.001 and runs < self.max_runs)):
            self.asses_fitness()
            self.update_velocity()
            self.update_positions()
            runs += 1
        print(runs)
            
            
