import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, data_types, seed):
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)
        self.step_notfall = 0
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_pendulum.xml'%self.dir_path, 2)
        if 'size' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_pendulum/inverted_pendulum_%s.xml'%(self.dir_path, self.data_type), 2)
            utils.EzPickle.__init__(self)
        elif 'mass' in self.data_type:
            self.original_mass = np.copy(self.model.body_mass)
            utils.EzPickle.__init__(self)
        elif 'damp' in self.data_type:
            self.original_damping = np.copy(self.model.dof_damping)
            utils.EzPickle.__init__(self)

        self._max_episode_steps = 1000
        self.fixed_data_type = False

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notfall = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        if not notfall:
            self.step_notfall = 0
        else:
            self.step_notfall += 1
        if self.step_notfall >= 150:
            reward = 1.0
            self.step_notfall = 0
        else:
            reward = 0.0
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)
        if 'size' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_pendulum/inverted_pendulum_%s.xml'%(self.dir_path, self.data_type), 2)
            utils.EzPickle.__init__(self)
            self.sim.reset() 
        elif 'mass' in self.data_type:
            self.mass_scale = float(self.data_type[4:])
            mass = np.copy(self.original_mass)
            mass *= self.mass_scale
            self.model.body_mass[:] = mass
        elif 'damp' in self.data_type:
            self.damping_scale = float(self.data_type[4:])
            damping = np.copy(self.original_damping)
            damping *= self.damping_scale
            self.model.dof_damping[:] = damping
        self.random_seed = self.rng.randint(100)
        self.seed(self.random_seed)
        self.step_notfall = 0
        return self.reset_model()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def get_sim_parameters(self):
        return float(self.data_type[4:])

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True
