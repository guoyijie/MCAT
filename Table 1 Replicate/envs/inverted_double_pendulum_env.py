import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedDoublePendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, data_types, seed):
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_double_pendulum.xml'%self.dir_path, 2)
        if 'size' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_double_pendulum/inverted_double_pendulum_%s.xml'%(self.dir_path, self.data_type), 5)
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
        x, _, y = self.sim.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.sim.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty 
        done = False
        return ob, r, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[:1],  # cart x pos
            np.sin(self.sim.data.qpos[1:]),  # link angles
            np.cos(self.sim.data.qpos[1:]),
            np.clip(self.sim.data.qvel, -10, 10),
            np.clip(self.sim.data.qfrc_constraint, -10, 10)
        ]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * .1
        )
        return self._get_obs()

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)
        if 'size' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_double_pendulum/inverted_double_pendulum_%s.xml'%(self.dir_path, self.data_type), 5)
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
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 0.5
        v.cam.lookat[2] = 0.12250000000000005  # v.model.stat.center[2]

    def get_sim_parameters(self):
        return float(self.data_type[4:])

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True
