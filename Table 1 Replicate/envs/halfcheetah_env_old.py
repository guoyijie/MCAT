import gym
import numpy as np

class HalfCheetahEnv(object):
    def __init__(self, data_types):
        self.armas = [float(x[4:]) for x in data_types]
        self.env = gym.make('HalfCheetah-v2')
        self.set_seed(0)
 
        self.arma = self.armas[0]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec        
        self._max_episode_steps = self.env._max_episode_steps

    def set_seed(self, seed):
        self.env.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def reset(self):
        self.arma = self.rng.choice(self.armas)
        self.env.unwrapped.model.dof_armature[3:] = self.arma
        ob = self.env.reset()
        return ob

    def step(self, action):
        ob, r, done, info = self.env.step(action)
        return ob, r, done, info

