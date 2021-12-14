import numpy as np
from gym.spaces import Box

class NormalizedEnv(object):

    def __init__(self,
                 env,
                 scale_reward=1.,
                 normalize_obs=False,
                 normalize_reward=False,
                 obs_alpha=0.001,
                 reward_alpha=0.001,
                 normalization_scale=1.,
                 dummy_flag=False,
                 ):

        self._scale_reward = 1
        self._wrapped_env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = env._max_episode_steps
        self.reward_range = env.reward_range
        self.metadata = env.metadata
        self.spec = env.spec
        self.set_seed = env.set_seed
        self.set_data_type = env.set_data_type

        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(self.observation_space.shape)
        self._obs_var = np.ones(self.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.
        self._normalization_scale = normalization_scale
        self._dummy_flag = dummy_flag

    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape) * self._normalization_scale
            return Box(-1 * ub, ub, dtype=np.float32)
        elif isinstance(self._wrapped_env.action_space, CustomBox):
            ub = np.ones(self._wrapped_env.action_space.shape) * self._normalization_scale
            return Box(-1 * ub, ub, dtype=np.float32)
        return self._wrapped_env.action_space

    def _update_obs_estimate(self, obs):
        o_a = self._obs_alpha
        self._obs_mean = (1 - o_a) * self._obs_mean + o_a * obs
        self._obs_var = (1 - o_a) * self._obs_var + o_a * np.square(obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        r_a = self._reward_alpha
        self._reward_mean = (1 - r_a) * self._reward_mean + r_a * reward
        self._reward_var = (1 - r_a) * self._reward_var + r_a * np.square(reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self):
        obs = self._wrapped_env.reset()
        if self._normalize_obs:
            return self._apply_normalize_obs(obs)
        else:
            return obs

    def step(self, action):
        if isinstance(self._wrapped_env.action_space, Box) or isinstance(self._wrapped_env.action_space, CustomBox):
            # rescale the action
            lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
            scaled_action = lb + (action + self._normalization_scale) * (ub - lb) / (2 * self._normalization_scale)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step

        if getattr(self, "_normalize_obs", False):
            print('normalize obs')
            next_obs = self._apply_normalize_obs(next_obs)
        if getattr(self, "_normalize_reward", False):
            print('normalize reward')
            reward = self._apply_normalize_reward(reward)
        return next_obs, reward * self._scale_reward, done, info

    def get_sim_parameters(self):
        if self._dummy_flag:
            original = self._wrapped_env.get_sim_parameters()
            original = np.zeros(original.shape)
            return original
        else:
            return self._wrapped_env.get_sim_parameters()
