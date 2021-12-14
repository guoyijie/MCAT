import math
from collections import OrderedDict

from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv
import numpy as np
import tensorflow as tf

from .base import EnvBinarySuccessMixin
from gym import error, spaces

################
### CARTPOLE ###
################
class ModifiableCartPoleEnv(CartPoleEnv, EnvBinarySuccessMixin):
    
    def _followup(self):
        """Cascade values of new (variable) parameters"""
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def reset(self):
        self.nsteps = 0
        return super(ModifiableCartPoleEnv, self).reset()
    
    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""
        self.nsteps += 1
        return super().step(*args, **kwargs)

    def is_success(self):

        """Returns True is current state indicates success, False otherwise
        Balance for at least 195 time steps ("definition" of success in Gym:
        https://github.com/openai/gym/wiki/CartPole-v0#solved-requirements)
        """
        target = 195
        
        if self.nsteps >= target:
            return True
        else:
            return False
    

class RandomCartPole_Force_Length(ModifiableCartPoleEnv):

    def __init__(self, data_types, seed):
        super(RandomCartPole_Force_Length, self).__init__()
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)

        self.force_mag = 10.0
        self.length = 0.5
        self.masspole = 0.1
       
        if 'forc' in self.data_type:
            self.force_mag = float(self.data_type[4:])
        elif 'leng' in self.data_type:
            self.length = float(self.data_type[4:])
 
        self._followup()
        self.fixed_data_type = False
        self._max_episode_steps = 1000

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)

        if 'forc' in self.data_type:
            self.force_mag = float(self.data_type[4:])
        elif 'leng' in self.data_type:
            self.length = float(self.data_type[4:])   
        
        self.masspole = 0.1

        self._followup()
        self.random_seed = self.rng.randint(100)
        self.seed(self.random_seed)
       
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
 
        return np.array(self.state)

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True 

    def get_sim_parameters(self):
        return float(self.data_type[4:])
