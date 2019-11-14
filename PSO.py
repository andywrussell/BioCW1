import numpy as np

class Particle:
    def __init__(self, position, velocity, ideal):
        #the index of a position or velocity in the list corresponds to a timestamp
        self.position_list = [position]
        self.velocity_list = [velocity]
        self.position = position
        self.velocity = velocity
        self.fittest = position
        self.previous_fittest = position
        self.ideal = ideal
        self.fitness = np.dot(self.position, self.ideal)
        self.previous_fitness = self.fitness
    
    def update_particle(self, new_pos, new_vel):
        self.update_position(new_pos)
        self.update_velocity(new_vel)

    def update_position(self, new_pos):
        self.position_list.append(new_pos)
        self.position = new_pos

    def update_velocity(self, new_vel):
        self.velocity_list.append(new_vel)
        self.velocity = new_vel

    def asses_fitness(self):
        current_pos = self.position_list[-1]
        fitness = np.dot(current_pos, self.ideal)
        if (self.fittest == []):
            self.fittest = current_pos
        
        #otherwise get distance from ideal and see if it is better than current best
        elif (fitness < self.best_fitness):
            self.previous_fittest = self.fittest
            self.previous_best_fitness = self.best_fitness
            self.fittest = current_pos
            self.best_fitness = fitness
                
    def informants_best(self):
        inf_best = self.informants[0].previous_fittest
        for inf in self.informants:
            if (inf.previous_best_fitness > inf_best.previous_best_fitness):
                inf_best = inf

        return inf_best


class PSO:
    def __init__ (self, swarmsize, alpha, beta, gamma, sigma, jumpsize, ideal, num_informants) :
        self.swarmsize = swarmsize #size of the swarm
        self.alpha = alpha #proportion of velocity to be retained
        self.beta = beta #proportion of personal best to be retained
        self.gamma = gamma #proportion of the 'informants' best to be retained
        self.sigma = sigma #proportion of global best to be retained
        self.jumpsize = jumpsize #jumpsize
        self.num_informants = num_informants #number of randomly selected informants per particle
        self.ideal = ideal

    def generate_particles(self):
        counter = 0
        self.particles = []
        while(counter < self.swarmsize):
            particle_pos = np.random.random_sample()
            particle_vel = np.random.random_sample()
            new_particle = Particle(particle_pos, particle_vel, ideal[counter])
            self.particles.append(new_particle)
            counter += 1

    def assign_informants(self):
        for particle in self.particles:
            particle.informants = []
            for i in range(self.num_informants):
                particle.informants.append(self.particles[np.random(0, self.swarmsize-1))

    def run_algo(self):
        self.asses_fitness()
        

    def asses_fitness(self):
        for particle in self.particles:
            particle.asses_fitness()
            if (!self.best or particle.fitness < self.best.fitness):
                self.previous_best = self.best
                self.best = particle
              
    def update_velocity(self):
        previous_b = self.previous_best
        
        for particle in self.particles:
            previous_f = particle.previous_fittest
            previous_inf = particle.informants_best()
            particle_vel = particle.velocity
            new_vel = []
            
            
            for i in range(len(particle_vel)):
                b = np.random(0, self.beta)
                c = np.random(0, self.gamma)
                d = np.random(0, self.sigma)
                new_vel.append( self.alpha*particle_vel[i] + b*(previous_f[i] - particle_vel[i]) + c(previous_inf[i] - particle_vel[i]) + d(previous_b[i] - particle_vel[i]))
                
                        
            particle.update_velocity(new_vel)
            
    def update_positions(self):
        for particle in particles:
            new_pos = particle.postition + particle.velocity
            particle.update_position(new_pos)
            






            