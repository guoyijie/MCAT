import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    # For SlimHumanoid, we followed the implementation from
    # https://github.com/WilsonWangTHU/mbbl/blob/master/mbbl/env/gym_env/humanoid.py
    def __init__(self, data_types, seed):
        self.set_seed(seed)
        self.data_types = data_types
        self.data_type = self.rng.choice(self.data_types) 
        
        self.prev_pos = None
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/humanoid.xml' % self.dir_path, 5)

        if 'arma' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/humanoid/humanoid_%s.xml'%(self.dir_path, self.data_type), 5)
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

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat])

    def step(self, a):
        old_obs = np.copy(self._get_obs())
        self.do_simulation(a, self.frame_skip)
        data = self.sim.data
        lin_vel_cost = 0.25 / 0.015 * old_obs[..., 22]
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        alive_bonus = 5.0 * (1 - float(done))
        done = False
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        pos_before = mass_center(self.model, self.sim)
        self.prev_pos = np.copy(pos_before)

        return self._get_obs()

    def reset(self):
        if not self.fixed_data_type:
            self.data_type = self.rng.choice(self.data_types)
        
        if 'arma' in self.data_type:
            mujoco_env.MujocoEnv.__init__(self, '%s/assets/humanoid/humanoid_%s.xml'%(self.dir_path, self.data_type), 5)
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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
   
    def get_sim_parameters(self):
        return float(self.data_type[4:])

    def set_data_type(self, data_type):
        self.data_type = data_type
        self.fixed_data_type = True

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]
