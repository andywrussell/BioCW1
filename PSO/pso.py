from PSO.particle import Particle
import numpy as np
from tqdm import tqdm

class PSO:
    def __init__ (self, net_generator, swarmsize, alpha, beta, gamma, delta, jumpsize, ideal, inputs, num_informants, max_runs, act_bound, weight_bound = 0, bound_strat = 3, vel_range = 1, informant_strat = 0):
        """
        Params
        ======
        * net_generator: Class that returns a neural net with given layers
        * swarmsize: size of swarm/number of particles for the PSO
        * alpha: proportion of current velocity to be retained when updating the velocity of a particle (typically 1)
        * beta: proportion of personal best to be retained when updating the velocity of a particle (typically 2.05)
        * gamma: proportion of the 'informants' best to be retained when updating the velocity of the particle (typically 2.05)
        * delta: proportion of global best to be retained when updating the velocity of the particle (typically 0)
        * jumpsize: the speed at which the particles move through the search space (typically 1)
        * num_informants: the number of informant particles to assign to each informant particle (typically 3).
            Only relevant when using informant_strat 0, 1 or 2
        * informant_strat: the strategy for initializing/updating the particle informants
            0 - Random initialization, informants stay static through PSO run
            1 - Random initialization, reinitialization after each iteration of algorithm
            2 - Random initialization, reinitialization after every iteration which fails to find a new global best
        * ideal: the desired output for the best particle
        * inputs: the inputs which will be fed to the ANN for each particle
        * max_runs: maximum number of runs before the PSO is terminated
        * best: the current best particle 
        * act_bound: the boundary limit for the activation functions of the ANN
        * weight_bound: the boundary limit for the weights of the ANN
        * bound_strat: strategy used to handle particles which exceed the specified boundary
            0: ignore boundary and allow particle to float away
            1: ignore move, retain current position inside violated dimension and assign new random velocity
            2: set particle position to the boundary it violated and reverse velocity
            3: bounce particle back into the search space by the distance which exceeded the boundary and reverse the velocity
        * vel_range: set the range for the initial random velocities (usually -1 to 1)
        """

        self.net_generator = net_generator
        self.swarmsize = swarmsize 
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma 
        self.delta = delta 
        self.jumpsize = jumpsize 
        self.num_informants = num_informants
        self.informant_strat = informant_strat
        self.ideal = ideal
        self.inputs = inputs
        self.max_runs = max_runs
        self.best = None 
        self.act_bound = act_bound 
        self.weight_bound = weight_bound 
        self.bound_strat = bound_strat 
        self.vel_range = vel_range
        self.unchanged_count = 0 #number iterations without finding a new global best
        self.unchanged_max = 100 #if unchanged_count count reaches unchanged_max terminate the algorithm

    def generate_particles(self):
        """
        Generates the number of particles according to the swarmsize
        New instance of neural net is added to each particle with random weights and activations
        """

        self.particles = []
        for i in range(self.swarmsize):
            network = self.net_generator.generate_network()
            network.flatten_net()

            particle_pos = network.net_as_vector 
            network.unflatten_net()
            particle_vel = np.array([])

            for j in range(len(particle_pos)):
                if (self.vel_range > 0):
                    particle_vel = np.append(particle_vel, np.random.uniform(-self.vel_range, self.vel_range))
                else:
                    particle_vel = np.append(particle_vel, 0)
                
            new_particle = Particle(network, particle_pos, particle_vel, self.ideal, self.inputs)
            self.particles.append(new_particle)


    def assign_informants(self):
        """
        Assigns informants to particles according the informant_strat
        """

        if (self.informant_strat == 3): #ring topology
            for i in range(0, self.swarmsize):
                self.particles[i].informants = []
                self.particles[i].informants.append(self.particles[i - 1 % self.swarmsize])
                if i == self.swarmsize -1:
                    self.particles[i].informants.append(self.particles[0])
                else:
                    self.particles[i].informants.append(self.particles[i + 1 % self.swarmsize])
        else:
            for particle in self.particles:
                particle.informants = []
                for i in range(self.num_informants):
                    rand_inf = np.random.choice(self.particles)
                    particle.informants.append(rand_inf)
                
    def asses_fitness(self):
        """
        Asses the fitness of each particle individual and update the global best as required
        If the best is unchanged increment unchanged_count and check the termination condition
        If appropriate reassign the informants based the selected informant_strat
        """

        best_changed = False
        for particle in self.particles:
            particle.asses_fitness()
            if self.best == None:
                best_changed = True
                self.best = Particle(particle.network, particle.position, particle.velocity, particle.ideal, particle.inputs)
            elif particle.fitness < self.best.fitness:
                best_changed = True
                self.best.network = particle.network
                self.best.update_position(particle.position)
                self.best.update_velocity(particle.velocity)
                self.best.fitness = particle.fitness
                self.best.best_fitness = particle.best_fitness
                self.best.best = particle.position
                self.best.best_list.append(particle.position)
                self.best.outputs = particle.outputs
        
        self.unchanged_count += 1
        if (best_changed):
            self.unchanged_count = 0

        # use adaptive random topology
        if self.informant_strat == 1:
            self.assign_informants()
        elif self.informant_strat == 2 and not best_changed:
            self.assign_informants()
                
    def update_velocity(self):
        """
        Update the velocity of each particle
        """ 
        best_pos = self.best.best_list[-2]  #get the previous best position                      
        
        for particle in self.particles:
            part_best = particle.best_list[-2]
            inf_best = particle.informants_best()
            particle_vel = particle.velocity
            new_vel = []

            b = np.random.uniform(0, self.beta)
            c = np.random.uniform(0, self.gamma)
            d = np.random.uniform(0, self.delta)
                        
            for i in range(len(particle_vel)):                
                b_val = b*(part_best[i] - particle.position[i]) 
                c_val = c*(inf_best[i] -particle.position[i])
                d_val = d*(best_pos[i] -particle.position[i])
                
                new_vel.append(self.alpha*particle_vel[i] + b_val + c_val + d_val)                
                        
            particle.update_velocity(new_vel)
            
    def update_positions(self):
        """
        Update the positions of each particle using the velocity according to the selected bound_strat
        """
        for particle in self.particles:
            new_pos = particle.position + (np.dot(self.jumpsize, particle.velocity)) #use dot to multiply jumpsize by velo array
            
            act_idx = particle.network.activation_idx
            for i in range(len(new_pos)):
                #if we are at an activation function impose boundary
                boundary = self.weight_bound
                is_activation = False
                if i in act_idx:
                    boundary = self.act_bound
                    is_activation = True
                    
                if boundary > 0:
                    if new_pos[i] > boundary or new_pos[i] < -boundary:
                        new_pos[i], particle.velocity[i] = self.enforce_boundary(new_pos[i], particle.position[i], particle.velocity[i], boundary, is_activation)

                         
            particle.update_position(new_pos)

    def enforce_boundary(self, new_pos, old_pos, vel, boundary, is_activation):
        """
        Use the boundary strategy to handle positions outside of the search space
        """
        if self.bound_strat == 0: #ignore boundary            
            return new_pos, vel

        elif self.bound_strat == 1: #ignore move
            return old_pos, np.random.uniform(-self.vel_range, self.vel_range)

        elif self.bound_strat == 2: #set to boundary
            if new_pos > boundary:
                new_pos = boundary
            elif new_pos < -boundary:
                new_pos = -boundary
            return new_pos, -vel

        elif self.bound_strat == 3: #bounce off boundary
            if new_pos > boundary:
                diff = new_pos - boundary
                new_pos = boundary - diff
            elif new_pos < -boundary:
                diff = abs(new_pos) - boundary
                new_pos = (-boundary) + diff
            return new_pos, -vel


    def run_algo(self):
        """
        Run the PSO to optimize the ANN for the selected values
        """
        self.generate_particles()
        self.assign_informants()
        run = 1

        progress_bar = tqdm(range(self.max_runs))
        for i in (progress_bar):
            if self.best == None or (self.best.fitness > 0.001):
                self.asses_fitness()
                self.update_velocity()
                self.update_positions()
                progress_bar.set_description(" Run {}/{} | Best fitness = {}".format(run, self.max_runs, round(self.best.fitness, 4)))
                run += 1
                if self.unchanged_count == self.unchanged_max:
                    break
            else:
                break            
            

            
            
