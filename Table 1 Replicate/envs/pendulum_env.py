import math
from collections import OrderedDict

from gym import spaces
from gym.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import tensorflow as tf

from .base import EnvBinarySuccessMixin
from gym import error, spaces

class ModifiablePendulumEnv(PendulumEnv):
    '''The pendulum environment without length and mass of object hard-coded.'''


    def __init__(self):
        super(ModifiablePendulumEnv, self).__init__()

        self.mass = 1.0
        self.length = 1.0
    
    def step(self, u):
        th, thdot = self.state
        g = 10.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        angle_normalize = ((th+np.pi) % (2*np.pi))-np.pi
        
        costs = angle_normalize**2 + .1*thdot**2 + .001*((u/2.0)**2) # original

        newthdot = thdot + (-3*g/(2*self.length) * np.sin(th + np.pi) + 3./(self.mass*self.length**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        normalized = ((newth+np.pi) % (2*np.pi))-np.pi

        self.state = np.array([newth, newthdot])

        # Extra calculations for is_success()
        # TODO(cpacker): be consistent in increment before or after func body
        self.nsteps += 1
        # Track how long angle has been < pi/3
        if -np.pi/3 <= normalized and normalized <= np.pi/3:
            self.nsteps_vertical += 1
        else:
            self.nsteps_vertical = 0
        # Success if if angle has been kept at vertical for 100 steps
        target = 100
        if self.nsteps_vertical >= target:
            #print("[SUCCESS]: nsteps is {}, nsteps_vertical is {}, reached target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = True
        else:
            #print("[NO SUCCESS]: nsteps is {}, nsteps_vertical is {}, target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = False

        return self._get_obs(), -costs, False, {}

    def reset(self, new=True):
        # Extra state for is_success()
        self.nsteps = 0
        self.nsteps_vertical = 0

        low = np.array([(7/8)*np.pi, -0.2])
        high = np.array([(9/8)*np.pi, 0.2])

        theta, thetadot = self.np_random.uniform(low=low, high=high)
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([theta, thetadot])

        self.last_u = None
        return self._get_obs()

    def is_success(self):
        """Returns True if current state indicates success, False otherwise
        Success: keep the angle of the pendulum at most pi/3 radians from
        vertical for the last 100 time steps of a trajectory with length 200
        (max_length is set to 200 in sunblaze_envs/__init__.py)
        """
        return self.success
    

class RandomPendulumAll(ModifiablePendulumEnv):
    
    def __init__(self, data_types, seed):

        super(RandomPendulumAll, self).__init__()    
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)

        if 'mass' in self.data_type:
            self.mass = float(self.data_type[4:])
        elif 'leng' in self.data_type:
            self.length = float(self.data_type[4:])

        self.fixed_data_type = False

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)

        if 'mass' in self.data_type:
            self.mass = float(self.data_type[4:])
        elif 'leng' in self.data_type:
            self.length = float(self.data_type[4:])

        self.random_seed = self.rng.randint(100)
        self.seed(self.random_seed)
        return super(RandomPendulumAll, self).reset()

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True

    def get_sim_parameters(self):
        return float(self.data_type[4:])
