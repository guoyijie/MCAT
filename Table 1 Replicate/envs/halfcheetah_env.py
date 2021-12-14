import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, data_types, seed=0):
        self.set_seed(seed)
        self.data_types = data_types
        self.raw_data_types = [float(x[4:]) for x in self.data_types]
        self.data_type = self.rng.choice(self.data_types)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if 'arma' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah/half_cheetah_%s.xml'%(self.dir_path, self.data_type), 5) 
            utils.EzPickle.__init__(self)
        elif 'mass' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml'%self.dir_path, 5)
            self.original_mass = np.copy(self.model.body_mass)
            utils.EzPickle.__init__(self, self.raw_data_types)
        elif 'damp' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml'%self.dir_path, 5)
            self.original_damping = np.copy(self.model.dof_damping)
            utils.EzPickle.__init__(self, self.raw_data_types)

        self._max_episode_steps = 1000
        self.fixed_data_type = False

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)
        if 'arma' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah/half_cheetah_%s.xml'%(self.dir_path, self.data_type), 5)
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
        return self.reset_model()
    
    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_sim_parameters(self):
        return float(self.data_type[4:])
