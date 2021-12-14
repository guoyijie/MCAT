import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, data_types, seed):
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/walker2d.xml'%self.dir_path, 4)
        if 'size' in self.data_type or 'arma' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/walker2d/walker2d_%s.xml'%(self.dir_path, self.data_type), 4)
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
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        return self._get_obs()

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)
        if 'size' in self.data_type or 'arma' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/walker2d/walker2d_%s.xml'%(self.dir_path, self.data_type), 4)
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

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def get_sim_parameters(self):
        return float(self.data_type[4:])

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True
