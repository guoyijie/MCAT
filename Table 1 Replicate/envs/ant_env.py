import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, data_types, seed):
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/ant.xml' % self.dir_path, 5)

        if 'mass' in self.data_type:        
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
        self.xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward_ctrl = -0.005 * np.square(a).sum()
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive

        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=reward_run,
            reward_ctrl=reward_ctrl,
            reward_contact=reward_contact,
            reward_survive=reward_survive)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])
    

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.xposbefore = self.get_body_com("torso")[0]
        return self._get_obs()

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)
        if 'mass' in self.data_type:
            self.mass_scale = float(self.data_type[4:])
            mass = np.copy(self.original_mass)
            mass[2:5] *= self.mass_scale
            mass[5:8] *= self.mass_scale
            mass[8:11] *= 1.0/self.mass_scale
            mass[11:14] *= 1.0/self.mass_scale
            self.model.body_mass[:] = mass
        elif 'damp' in self.data_type:
            self.damping_scale = float(self.data_type[4:])
            damping = np.copy(self.original_damping)
            damping[2:5] *= self.damping_scale
            damping[5:8] *= self.damping_scale
            damping[8:11] *= 1.0/self.damping_scale
            damping[11:14] *= 1.0/self.damping_scale
            self.model.dof_damping[:] = damping

        self.random_seed = self.rng.randint(100)
        self.seed(self.random_seed)
        return self.reset_model()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
   
    def get_sim_parameters(self):
        return float(self.data_type[4:])

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True
