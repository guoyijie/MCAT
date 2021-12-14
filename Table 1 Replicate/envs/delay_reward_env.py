import gym
import numpy as np

class DelayRewardEnv(object):
    def __init__(self, env, reward_delay=1):
        self.env = env

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.spec = self.env.spec
        self._max_episode_steps = self.env._max_episode_steps

        self.t = 0
        self.cum_reward = 0
        self.reward_delay = reward_delay
        self.set_seed = self.env.set_seed        
        self.get_sim_parameters = self.env.get_sim_parameters
        self.set_data_type = self.env.set_data_type

    def reset(self):
        self.t = 0
        self.cum_reward = 0
        ob = self.env.reset()
        return ob


    def step(self, action):
        ob, r, done, info = self.env.step(action)
        self.cum_reward += r
        self.t += 1
        if self.t%self.reward_delay==0 or done:
            r = self.cum_reward
            self.cum_reward = 0
        else:
            r = 0
        return ob, r, done, info

    def get_sim_parameters(self):
        return np.array([float(self.env.data_type[4:])])
