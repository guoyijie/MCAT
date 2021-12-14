import numpy as np
import os
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, data_types, seed):
        """
        If extreme set=[0], neutral
        If extreme set=[1], extreme
        """
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types)
 
        self.prev_qpos = None

        self.cripple_mask = None

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml'%self.dir_path, 5)

        self.cripple_mask = np.ones(self.action_space.shape)
        self.cripple_set = []
        for data_type in self.data_types:
            x = data_type[4:].split(',')
            self.cripple_set += list(map(int, x)) 
        self.cripple_set = list(set(self.cripple_set))
 
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

        utils.EzPickle.__init__(self, self.cripple_set, [0,1])
        self._max_episode_steps = 1000

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        if self.cripple_mask is None:
            action = action
        else:
            action = self.cripple_mask * action
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()

        reward_ctrl = -0.1  * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_run + reward_ctrl

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)

        return self._get_obs()

    def reset(self):
        action_dim = self.action_space.shape
        x = self.data_type[4:].split(',') 
        self.crippled_joint = np.array(list(map(int, x)))
        self.cripple_mask = np.ones(action_dim)
        self.cripple_mask[self.crippled_joint] = 0

        geom_rgba = self._init_geom_rgba.copy()
        for joint in self.crippled_joint:
            geom_idx = self.model.geom_names.index(self.model.joint_names[joint+3])
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba.copy()
        self.random_seed = self.rng.randint(100)
        self.seed(self.random_seed)
        return self.reset_model()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
    
    def get_sim_parameters(self):
        return int(self.data_type[4:])

    def set_data_type(self, data_type):
        self.data_type = data_type
