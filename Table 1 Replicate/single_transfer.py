from models import ContextModel
from models import ContextForwardModel, ImagePool, MultiPairAxmodel, PairAxmodel, ConditionDmodel, GANLoss
import torch.nn as nn
from collections import OrderedDict, deque
import numpy as np
import time
import joblib
from plot_utils import context_tsne_plot, improvement_plot
import torch
import torch.nn.functional as F
import tensor_utils
import copy

class CycleData(object):
    def __init__(self, env, data_type, seed, history_length, sample_max=1e6):
        self.data_type = data_type
        self._dataset = None
        self.sample_n = 0
        self.sample_max = int(sample_max)
        self.rng = np.random.RandomState(seed)

    def sample(self, batch_size=256):
        if self.sample_n>0:
            batch_idx = self.rng.random_integers(low=0, high=self.sample_n-1, size=batch_size)
            obs = self._dataset['obs'][batch_idx]
            act = self._dataset['act'][batch_idx]
            delta = self._dataset['delta'][batch_idx]
            cp_obs = self._dataset['cp_obs'][batch_idx]
            cp_act = self._dataset['cp_act'][batch_idx]
            obs_next = self._dataset['obs_next'][batch_idx]
            future_bool = self._dataset['future_bool'][batch_idx]
            sim_params = self._dataset["sim_params"][batch_idx]
            sample = (obs, act, delta, obs_next, cp_obs, cp_act, future_bool, sim_params)
        else:
            sample = (None, None, None, None, None, None, None, None)
        return sample

    def sample_single(self, batch_size=1024):
        batch_idx = self.rng.random_integers(low=0, high=self.sample_n-1, size=batch_size)
        obs, act, next_obs, cp_obs, cp_act =  self._dataset['single_obs'][batch_idx], self._dataset['single_act'][batch_idx], self._dataset['single_delta'][batch_idx]+self._dataset['single_obs'][batch_idx], self._dataset['cp_obs'][batch_idx], self._dataset['cp_act'][batch_idx]
        return self.to_device(obs), self.to_device(act), self.to_device(next_obs), self.to_device(cp_obs), self.to_device(cp_act)

    def to_device(self,data):
        if torch.cuda.is_available():
            return torch.tensor(data).float().cuda()
        else:
            return torch.tensor(data).float()

    def add(self, obs, act, delta, cp_obs, cp_act, future_bool, obs_next, sim_params, single_obs, single_act, single_delta):
        if self._dataset is None:
            self._dataset = dict(obs=obs, act=act, delta=delta,
                                 cp_obs=cp_obs, cp_act=cp_act, future_bool=future_bool,
                                 obs_next=obs_next,
                                 sim_params=sim_params,
                                 single_obs=single_obs, single_act=single_act,
                                 single_delta=single_delta)
        else:
            self._dataset['obs'] = np.concatenate([self._dataset['obs'], obs])
            self._dataset['act'] = np.concatenate([self._dataset['act'], act])
            self._dataset['delta'] = np.concatenate([self._dataset['delta'], delta])
            self._dataset['cp_obs'] = np.concatenate([self._dataset['cp_obs'], cp_obs])
            self._dataset['cp_act'] = np.concatenate([self._dataset['cp_act'], cp_act])
            self._dataset['future_bool'] = np.concatenate([self._dataset['future_bool'], future_bool])
            self._dataset['obs_next'] = np.concatenate([self._dataset['obs_next'], obs_next])
            self._dataset["sim_params"] = np.concatenate([self._dataset["sim_params"], sim_params])
            self._dataset['single_obs'] = np.concatenate([self._dataset['single_obs'], single_obs])
            self._dataset['single_act'] = np.concatenate([self._dataset['single_act'], single_act])
            self._dataset['single_delta'] = np.concatenate([self._dataset['single_delta'], single_delta])
        self.sample_n += obs.shape[0]
        assert(self._dataset['obs'].shape[0]==self.sample_n)
        assert(self._dataset['sim_params'].shape[0]==self.sample_n)

        if self.sample_n > self.sample_max:
            self._dataset['obs'] = self._dataset['obs'][self.sample_n-self.sample_max:]
            self._dataset['act'] = self._dataset['act'][self.sample_n-self.sample_max:]
            self._dataset['delta'] = self._dataset['delta'][self.sample_n-self.sample_max:]
            self._dataset['cp_obs'] = self._dataset['cp_obs'][self.sample_n-self.sample_max:]
            self._dataset['cp_act'] = self._dataset['cp_act'][self.sample_n-self.sample_max:]
            self._dataset['future_bool'] = self._dataset['future_bool'][self.sample_n-self.sample_max:]
            self._dataset['obs_next'] = self._dataset['obs_next'][self.sample_n-self.sample_max:]
            self._dataset["sim_params"] = self._dataset["sim_params"][self.sample_n-self.sample_max:]
            self._dataset['single_obs'] = self._dataset['single_obs'][self.sample_n-self.sample_max:]
            self._dataset['single_act'] = self._dataset['single_act'][self.sample_n-self.sample_max:]
            self._dataset['single_delta'] = self._dataset['single_delta'][self.sample_n-self.sample_max:]
        self.sample_n = self._dataset['obs'].shape[0]

class ActionTransferAgent(object):
    """
    Class for MLP continous dynamics model
    """

    # def __init__(self,
    #              transfer_envs,
    #              data_types,
    #              batch_size=256,
    #              forward_n_update=5000,
    #              cycle_n_update=5000,
    #              context_out_dim=10,
    #              history_length=10,
    #              future_length=10,
    #              state_diff=True,
    #              weight_decay_coeff=1.0,
    #              deterministic=False,
    #              forward_lr=1e-3,
    #              constractive_weight=0,
    #              cycle_weight = 1,
    #              back_weight = 0,
    #              identity_weight = 0,
    #              threshold =0
    #              ):


    #     # Default Attributes
    #     self.transfer_envs = transfer_envs
    #     self.data_types=data_types
    #     self.task_nums = len(self.data_types)
    #     self.transfer_source_idx = [None for _ in range(self.task_nums)]

    #     self.nested_agents = [None for _ in range(self.task_nums)]
    #     for j in range(self.task_nums):
    #         self.nested_agents[j] = CycleData(seed=0)


    #     # Dynamics Model Attributes
    #     self.state_diff = state_diff
    #     self.weight_decay_coeff=weight_decay_coeff
    #     self.constractive_weight=constractive_weight
    #     self.deterministic = deterministic

    #     self.normalize_input = True
    #     self.batch_size = batch_size

    #     self.context_out_dim = context_out_dim
    #     self.history_length = history_length
    #     self.future_length = future_length
    #     self.forward_n_update = forward_n_update
    #     self.cycle_n_update = cycle_n_update
    #     self.cycle_weight = cycle_weight
    #     self.back_weight = back_weight
    #     self.identity_weight = identity_weight
    #     self.threshold = threshold

    #     # Dimensionality of state and action space
    #     self.obs_space_dims = obs_space_dims = 14
    #     self.action_space_dims = action_space_dims = 4

    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     self.cp = ContextModel(context_dim=(obs_space_dims+action_space_dims)*self.history_length, context_out_dim=self.context_out_dim, device=self.device)
            
    #     self.mlp = ContextForwardModel(input_obs_dim=obs_space_dims, input_act_dim=action_space_dims, context_out_dim=self.context_out_dim, output_dim=obs_space_dims, device=self.device)
        

    #     self.model = PairAxmodel(state_dim=obs_space_dims, action_dim=action_space_dims).to(self.device)
    #     self.back_model = PairAxmodel(state_dim=obs_space_dims, action_dim=action_space_dims).to(self.device)
 
    #     lr = 3e-4
    #     self.optimizer_g = torch.optim.Adam([{'params': self.back_model.parameters(), 'lr': lr}, {'params': self.model.parameters(), 'lr': lr}])
    #     self.optimizer = self.optimizer_f = torch.optim.Adam([{'params': self.cp.parameters(), 'lr': forward_lr}, {'params': self.mlp.parameters(), 'lr': forward_lr}]) 

    #     self.initialize_running_path()
    #     if self.normalize_input:
    #         self.normalization = {'obs':create_normalization(self.obs_space_dims),
    #                               'act':create_normalization(self.action_space_dims),
    #                               'delta':create_normalization(self.obs_space_dims),
    #                               'cp_obs':create_normalization(self.obs_space_dims*self.history_length),
    #                               'cp_act':create_normalization(self.action_space_dims*self.history_length),
    #                               }
    #     self.forward_loss = 10
    def __init__(self,
                 env_name, 
                 env,
                 data_types,
                 seed,
                 batch_size=256,
                 forward_n_update=5000,
                 cycle_n_update=5000,
                 context_out_dim=10,
                 history_length=10,
                 future_length=10,
                 state_diff=True,
                 weight_decay_coeff=1.0,
                 constractive_weight=1.0,
                 deterministic=False,
                 eval_n=5,
                 forward_lr=1e-3
                 ):

        # Default Attributes
        self.env = env
        self.env_name = env_name
        self.data_types = data_types
        self.task_nums = len(self.data_types)
        self.raw_data_types = [float(data_type[4:]) for data_type in data_types]

        self.nested_agents = [None for _ in data_types]
        for j, target in enumerate(data_types):
            self.nested_agents[j] = CycleData(env, seed=seed, data_type=[target], history_length=history_length)

        self.real_pools = [ImagePool(256, seed=seed) for _ in range(len(data_types))]
        self.fake_pools = [ImagePool(256, seed=seed) for _ in range(len(data_types))]

        self._dataset = None

        # Dynamics Model Attributes
        self.state_diff = state_diff
        self.weight_decay_coeff=weight_decay_coeff
        self.constractive_weight=constractive_weight
        self.deterministic = deterministic

        self.normalize_input = True
        self.batch_size = batch_size

        self.context_out_dim = context_out_dim
        self.history_length = history_length
        self.future_length = future_length
        self.forward_n_update = forward_n_update
        self.cycle_n_update = cycle_n_update

        # Dimensionality of state and action space
        self.obs_space_dims = obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = action_space_dims = env.action_space.shape[0]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cp = ContextModel(context_dim=(obs_space_dims+action_space_dims)*self.history_length, context_out_dim=self.context_out_dim, device=self.device)
            
        self.mlp = ContextForwardModel(input_obs_dim=obs_space_dims, input_act_dim=action_space_dims, context_out_dim=self.context_out_dim, output_dim=obs_space_dims, device=self.device)
        
        self.model = PairAxmodel(state_dim=obs_space_dims, action_dim=action_space_dims).to(self.device)
        self.back_model = PairAxmodel(state_dim=obs_space_dims, action_dim=action_space_dims).to(self.device)
        # self.model = MultiPairAxmodel(state_dim=obs_space_dims, action_dim=action_space_dims, context_dim=context_out_dim).to(self.device)
        # self.back_model = MultiPairAxmodel(state_dim=obs_space_dims, action_dim=action_space_dims, context_dim=context_out_dim).to(self.device)
        self.dmodel = ConditionDmodel(state_dim=obs_space_dims, action_dim=action_space_dims, context_dim=context_out_dim).to(self.device)
        self.criterionGAN = GANLoss().to(self.device)
  
        lr = 3e-4
        self.optimizer_g = torch.optim.Adam([{'params': self.back_model.parameters(), 'lr': lr}, {'params': self.model.parameters(), 'lr': lr}])
        self.optimizer_d = torch.optim.Adam(self.dmodel.parameters(),lr=lr) 
        self.optimizer = self.optimizer_f = torch.optim.Adam([{'params': self.cp.parameters(), 'lr': forward_lr}, {'params': self.mlp.parameters(), 'lr': forward_lr}]) 

        self.initialize_running_path()
        if self.normalize_input:
            self.normalization = {'obs':create_normalization(self.obs_space_dims),
                                  'act':create_normalization(self.action_space_dims),
                                  'delta':create_normalization(self.obs_space_dims),
                                  'cp_obs':create_normalization(self.obs_space_dims*self.history_length),
                                  'cp_act':create_normalization(self.action_space_dims*self.history_length),
                                  }
        self.forward_loss = 10

    def initialize_running_path(self):
        self.running_path = {}
        self.running_path['observations'] = []
        self.running_path['actions'] = []
        self.running_path['cp_obs'] = []
        self.running_path['cp_act'] = []
        self.running_path['sim_params'] = []
        self.running_sim_param = None

    def add(self, obs, action, done, sim_param, cp_obs, cp_act):
        if self.running_sim_param is not None:
            assert sim_param == self.running_sim_param
        self.running_sim_param = sim_param
        self.running_path['observations'].append(copy.deepcopy(obs))
        self.running_path['actions'].append(copy.deepcopy(action))
        self.running_path['cp_obs'].append(copy.deepcopy(cp_obs))
        self.running_path['cp_act'].append(copy.deepcopy(cp_act))
        self.running_path['sim_params'].append(copy.deepcopy(sim_param))
        if done:
            path_len = len(self.running_path['observations'])
            self.running_path['observations'] = np.array(self.running_path['observations'])
            self.running_path['actions'] = np.array(self.running_path['actions'])
            self.running_path['cp_obs'] = np.array(self.running_path['cp_obs']).reshape((path_len, -1))
            self.running_path['cp_act'] = np.array(self.running_path['cp_act']).reshape((path_len, -1))
            self.running_path['sim_params'] = np.array(self.running_path['sim_params'])
            #task_idx = self.raw_data_type.index(sim_param)
            #seg = self.process_traj(self.running_path)
            #self.nested_agents[context_index].add(seg)
            samples_data = self.process_paths([self.running_path])
            self.add_samples(self.raw_data_types.index(self.running_sim_param),
                           samples_data['concat_obs'],
                           samples_data['concat_act'],
                           samples_data['concat_next_obs'],
                           samples_data["sim_params"],
                           samples_data['cp_observations'],
                           samples_data['cp_actions'],
                           samples_data['concat_bool']
                           )
            self.initialize_running_path()


    def process_paths(self, paths):
        obs_dim = paths[0]["observations"].shape[1]
        act_dim = paths[0]["actions"].shape[1]
        cp_obs_dim = paths[0]["cp_obs"].shape[1]
        cp_act_dim = paths[0]["cp_act"].shape[1]
        recurrent = False

        concat_obs_list, concat_act_list, concat_next_obs_list, concat_bool_list= [], [], [], []
        for path in paths:
            path_len = path["observations"].shape[0]
            remainder = 0
            if path_len < self.future_length + 1:
                remainder = self.future_length + 1 - path_len
                path["observations"] = np.concatenate([path["observations"], np.zeros((remainder, obs_dim))], axis=0)
                path["actions"] = np.concatenate([path["actions"], np.zeros((remainder, act_dim))], axis=0)
                path["cp_obs"] = np.concatenate([path["cp_obs"], np.zeros((remainder, cp_obs_dim))], axis=0)
                path["cp_act"] = np.concatenate([path["cp_act"], np.zeros((remainder, cp_act_dim))], axis=0)
                path['sim_params'] = np.concatenate((path['sim_params'], np.ones((remainder, ))*path['sim_params'][-1]))
            concat_bool = np.ones((path["observations"][:-1].shape[0], self.future_length))
            for i in range(self.future_length):
                if i == 0:
                    concat_obs = path["observations"][:-1]
                    concat_act = path["actions"][:-1]
                    concat_next_obs = path["observations"][1:]
                    temp_next_act = path["actions"][1:]
                else:
                    temp_next_obs = np.concatenate([path["observations"][1+i:], np.zeros((i, obs_dim))], axis=0)
                    concat_obs = np.concatenate([concat_obs, concat_next_obs[:, -obs_dim:]], axis=1)
                    concat_next_obs = np.concatenate([concat_next_obs, temp_next_obs], axis=1)
                    concat_act = np.concatenate([concat_act, temp_next_act], axis=1)
                    temp_next_act = np.concatenate([path["actions"][1+i:], np.zeros((i, act_dim))], axis=0)

                start_idx = max(i - remainder, 0)
                concat_bool[-i][start_idx:] = 0

            concat_obs_list.append(concat_obs)
            concat_act_list.append(concat_act)
            concat_next_obs_list.append(concat_next_obs)
            concat_bool_list.append(concat_bool)
        concat_next_obs_list = tensor_utils.concat_tensor_list(concat_next_obs_list, recurrent)
        concat_obs_list = tensor_utils.concat_tensor_list(concat_obs_list, recurrent)
        concat_act_list = tensor_utils.concat_tensor_list(concat_act_list, recurrent)
        concat_bool_list = tensor_utils.concat_tensor_list(concat_bool_list, recurrent)

        cp_observations = tensor_utils.concat_tensor_list([path["cp_obs"][:-1] for path in paths], recurrent)
        cp_actions = tensor_utils.concat_tensor_list([path["cp_act"][:-1] for path in paths], recurrent)
        sim_params_dynamics = tensor_utils.concat_tensor_list([path["sim_params"][:-1] for path in paths], recurrent)
        return dict(
                cp_observations=cp_observations,
                cp_actions=cp_actions,
                concat_next_obs=concat_next_obs_list,
                concat_obs=concat_obs_list,
                concat_act=concat_act_list,
                concat_bool=concat_bool_list,
                sim_params=sim_params_dynamics,
            )


    def add_samples(self, idx, obs, act, obs_next, sim_params, cp_obs, cp_act, future_bool, compute_normalization=True):
        assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims*self.future_length
        assert obs_next.ndim == 2 and obs_next.shape[1] == self.obs_space_dims*self.future_length
        assert act.ndim == 2 and act.shape[1] == self.action_space_dims*self.future_length
        assert cp_obs.ndim == 2 and cp_obs.shape[1] == (self.obs_space_dims*self.history_length)
        assert cp_act.ndim == 2 and cp_act.shape[1] == (self.action_space_dims*self.history_length)
        assert future_bool.ndim == 2 and future_bool.shape[1] == self.future_length

        sim_params = sim_params.reshape(-1, 1)
        obs = obs.reshape(-1, self.obs_space_dims)
        obs_next = obs_next.reshape(-1, self.obs_space_dims)
        delta = obs_next - obs

        obs = obs.reshape(-1, self.future_length * self.obs_space_dims)
        obs_next = obs_next.reshape(-1, self.future_length * self.obs_space_dims)
        delta = delta.reshape(-1, self.future_length * self.obs_space_dims)

        single_obs = obs[:, :self.obs_space_dims]
        single_act = act[:, :self.action_space_dims]
        single_delta = delta[:, :self.obs_space_dims]

        self.nested_agents[idx].add(obs, act, delta, cp_obs, cp_act, future_bool, obs_next, sim_params, single_obs, single_act, single_delta)

        self.compute_normalization(single_obs,
                                   single_act,
                                   single_delta,
                                   cp_obs,
                                   cp_act,
                                   )
    
    
    def get_forward_loss(self, idx):
        (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std) = self.get_normalization_stats()

        train_obs, train_act, train_delta, train_obs_next, train_cp_obs, train_cp_act, train_future_bool, train_sim_param= self.nested_agents[idx].sample(256) 
        train_obs, train_act, train_delta, train_obs_next, train_cp_obs, train_cp_act, train_sim_param= \
            self._preprocess_inputs(train_obs, train_act, train_delta, train_cp_obs, train_cp_act, train_future_bool, train_obs_next, train_sim_param)

        bootstrap_batch_idx = np.random.randint(train_obs.shape[0], size=(256))
        bootstrap_train_obs = torch.FloatTensor(train_obs[bootstrap_batch_idx]).to(self.device)
        bootstrap_train_act = torch.FloatTensor(train_act[bootstrap_batch_idx]).to(self.device)
        bootstrap_train_delta = torch.FloatTensor(train_delta[bootstrap_batch_idx]).to(self.device)
        bootstrap_train_sim_param = train_sim_param[bootstrap_batch_idx]
        bootstrap_train_obs_next = torch.FloatTensor(train_obs_next[bootstrap_batch_idx]).to(self.device)
        bootstrap_train_cp_obs = torch.FloatTensor(train_cp_obs[bootstrap_batch_idx]).to(self.device)
        bootstrap_train_cp_act = torch.FloatTensor(train_cp_act[bootstrap_batch_idx]).to(self.device)

        context = self.cp(bootstrap_train_cp_obs, bootstrap_train_cp_act, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std)
        _, mu, logvar = self.mlp(bootstrap_train_obs, bootstrap_train_act, context, norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, self.deterministic)
        normalized_delta = normalize(bootstrap_train_delta, norm_delta_mean, norm_delta_std)
        mse_loss = torch.sum(torch.mean(torch.mean(torch.square(mu - normalized_delta), dim=-1), dim=-1))
        l2_loss = self.cp.l2_reg()+self.mlp.l2_reg()
        if self.deterministic:
            recon_loss = mse_loss
            loss = mse_loss + l2_loss * self.weight_decay_coeff
        else:
            invvar = torch.exp(-logvar)
            mu_loss = torch.sum(torch.mean(torch.mean(torch.square(mu - normalized_delta) * invvar, dim=-1), dim=-1))
            var_loss = torch.sum(torch.mean(torch.mean(logvar, dim=-1), dim=-1))
            recon_loss = mse_loss + var_loss
            reg_loss = 0.01 * torch.sum(self.mlp.max_logvar) - 0.01 * torch.sum(self.mlp.min_logvar)
            loss = mu_loss + var_loss + reg_loss + l2_loss * self.weight_decay_coeff
        return context, bootstrap_train_sim_param, mse_loss, recon_loss, loss


    def train_context_forward(self, writer=None, t=0):

        learned_context = np.array([], np.float32).reshape((0, self.context_out_dim))
        given_task_idx = np.array([], np.float32).reshape((0, 1))
        if self.env_name=='HalfCheetah-v2' and self.forward_loss < 0.01:
            forward_n_update = int(1000)
        elif self.env_name=='Ant-v2' and self.forward_loss < 0.2:
            forward_n_update = int(1000)
        elif self.env_name=='Humanoid-v2' and self.forward_loss < 0.45: #0.2:
            forward_n_update = int(1000)
        elif self.env_name=='Hopper-v2' and self.forward_loss < 0.1:
            forward_n_update = int(1000)
        elif self.env_name=='Walker2d-v2' and self.forward_loss <0.2: #< 0.05:
            forward_n_update = int(1000)
        else:
            forward_n_update = self.forward_n_update
        mse_losses, recon_losses, const_losses = [], [], []
        for batch_num in range(forward_n_update):
            #if np.random.rand() < 0.5:
            #    a_idx = np.random.randint(0, self.task_nums)
            #    b_idx = a_idx
            #    label = 0
            #else:
            #    a_idx, b_idx = np.random.choice(list(np.arange(self.task_nums)), size=2, replace=False)
            #    assert not a_idx==b_idx
            #    label = 1
            a_idx = 1
            b_idx = 1
            label = 0
            a_context, a_sim_param, a_mse_loss, a_recon_loss, a_loss = self.get_forward_loss(a_idx)
            b_context,b_sim_param, b_mse_loss, b_recon_loss, b_loss = self.get_forward_loss(b_idx)
            loss = a_loss + b_loss
            if self.constractive_weight > 0:
                euclidean_dis = F.pairwise_distance(a_context, b_context)
                label_tensor = torch.ones(a_context.shape[0]).to(self.device)*label
                MARGIN = 1.0
                contrastive_loss = (1-label_tensor)*torch.pow(euclidean_dis,2) + label_tensor * torch.pow(torch.clamp(MARGIN-euclidean_dis,min=0),2)
                contrastive_loss = contrastive_loss.mean()*self.constractive_weight
                loss += contrastive_loss
                const_losses.append(contrastive_loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            mse_losses.append(a_mse_loss.item()+b_mse_loss.item())
            recon_losses.append(a_recon_loss.item()+b_recon_loss.item())
            if batch_num>forward_n_update-10:
                context = a_context.cpu().data.numpy()
                learned_context = np.concatenate((learned_context, context), axis=0)
                given_task_idx = np.concatenate((given_task_idx, np.ones((a_context.shape[0], 1))*a_idx), axis=0)
            if batch_num%100 == 0:
               print("Training DynamicsModel - finished epoch %d --"
                               "[Training] mse loss: %.4f  recon loss: %.4f  const loss: %.4f"
                                % (batch_num, np.mean(mse_losses), np.mean(recon_losses), np.mean(const_losses)))

        writer.add_scalar('forward_loss', np.mean(mse_losses), t)
        writer.add_scalar('constractive_loss', np.mean(const_losses), t)
        self.forward_loss = np.mean(mse_losses)

    def get_optim(self,lr):
        optimizer_g = torch.optim.Adam([{'params': self.back_model.parameters(), 'lr': lr}, {'params': self.model.parameters(), 'lr': lr}])
        return optimizer_g

    def train_transfer_agent_bisimilar(self, writer=None, t=0):
        if t < 5:
            self.optimizer_g = self.get_optim(3e-4)
        # else:
        # 	self.optimizer_g = self.get_optim(1e-5)
        elif t < 15:
        	self.optimizer_g = self.get_optim(5e-5)
        else:
        	self.optimizer_g = self.get_optim(1e-5)
        (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std) = self.get_normalization_stats()

        loss_fn = nn.L1Loss()
        epoch_loss, cmp_loss, gan_loss = [], [], []
        # epoch_cycle_loss, epoch_identity_loss, epoch_back_loss = [], [], []
        pair_transfer_distance = [[[] for _ in range(self.task_nums)] for _ in range(self.task_nums)]
        target_context_tensor =None
        for i in range(self.cycle_n_update):
            #source_index, target_index = np.random.choice(list(np.arange(self.task_nums)), size=2)
            source_index, target_index = 0, 1
            source_agent = self.nested_agents[source_index]
            target_agent = self.nested_agents[target_index]
            source_sim_param = float(self.data_types[source_index][4:])
            target_sim_param = float(self.data_types[target_index][4:])

            item1 = self.nested_agents[source_index].sample_single(batch_size=self.batch_size)
            if item1[0] is None:
                continue
            source_state, source_action, source_nxt_state, cp_obs, cp_act = item1
            
            source_context_tensor = self.cp(cp_obs, cp_act, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std).detach()
            mean_source_context_tensor = torch.mean(source_context_tensor, dim=0, keepdim=True)
            mean_source_context_tensor = mean_source_context_tensor.repeat((source_action.shape[0], 1))

            item2 = self.nested_agents[target_index].sample_single(batch_size=self.batch_size)
            if item2[0] is None:
                continue

            target_state, target_action, target_nxt_state, cp_obs, cp_act = item2
            target_context_tensor = self.cp(cp_obs, cp_act, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std).detach()
            mean_target_context_tensor = torch.mean(target_context_tensor, dim=0, keepdim=True)
            mean_target_context_tensor = mean_target_context_tensor.repeat((target_action.shape[0], 1))
            trans_action = self.model(source_state, source_action)

            _, out_mu, out_logvar = self.mlp(source_state, trans_action, mean_target_context_tensor, norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, self.deterministic)
            delta = source_nxt_state-source_state
            normalized_delta = normalize(delta, norm_delta_mean, norm_delta_std)
            loss_cycle = nn.L1Loss()(normalized_delta, out_mu)

            loss_all = loss_cycle
            # loss_identity = nn.L1Loss()(trans_action, source_action)
    
            # back_action = self.back_model(source_state, trans_action)
            # loss_back = nn.L1Loss()(source_action, back_action)

            # loss_all = loss_cycle*self.cycle_weight + loss_back*self.back_weight + loss_identity*self.identity_weight

            self.optimizer_g.zero_grad()
            loss_all.backward()
            self.optimizer_g.step()

            # epoch_cycle_loss.append(loss_cycle.item())
            # epoch_identity_loss.append(loss_identity.item())
            # epoch_back_loss.append(loss_back.item())
            epoch_loss.append(loss_cycle.item())
            pair_transfer_distance[source_index][target_index].append(loss_cycle.item())
            if i%100 == 0:
                print('cycle_loss:{:.3f}'.format(np.mean(epoch_loss)))

        # writer.add_scalar('cycle_loss', np.mean(epoch_cycle_loss), t)
        # writer.add_scalar('identity_loss', np.mean(epoch_identity_loss), t)
        # writer.add_scalar('back_loss', np.mean(epoch_back_loss), t)
        writer.add_scalar('cycle_loss', np.mean(epoch_loss), t)
        for i in range(self.task_nums):
            for j in range(self.task_nums):
                name = '%d-%d'%(i, j)
                writer.add_scalar('%s/transfer_distance'%name, np.mean(pair_transfer_distance[i][j]), t)


    def update_normalization(self, name, data):
        self.normalization[name]['sum'] += np.sum(data, axis=0)
        self.normalization[name]['square_sum'] += np.sum(data**2, axis=0)
        self.normalization[name]['n'] += data.shape[0]
        self.normalization[name]['mean'] = self.normalization[name]['sum']/self.normalization[name]['n']
        tmp = self.normalization[name]['square_sum'] - 2*self.normalization[name]['sum']*self.normalization[name]['mean'] + self.normalization[name]['n']*(self.normalization[name]['mean'])**2
        self.normalization[name]['std'] = np.sqrt(tmp/self.normalization[name]['n'])
        self.normalization[name]['mean'] = np.array(self.normalization[name]['mean'], np.float32)
        self.normalization[name]['std'] = np.array(self.normalization[name]['std'], np.float32)

    def compute_normalization(self, obs, act, delta, cp_obs, cp_act): 
        assert obs.shape[0] == delta.shape[0] == act.shape[0]

        # store means and std in dict
        self.update_normalization('obs', obs)
        self.update_normalization('act', act)
        self.update_normalization('delta', delta)
        self.update_normalization('cp_obs', cp_obs)
        self.update_normalization('cp_act', cp_act)

    def get_normalization_stats(self):
        if self.normalize_input:
            norm_obs_mean = self.normalization['obs']['mean']
            norm_obs_std = self.normalization['obs']['std']
            norm_delta_mean = self.normalization['delta']['mean']
            norm_delta_std = self.normalization['delta']['std']
            norm_act_mean = self.normalization['act']['mean']
            norm_act_std = self.normalization['act']['std']
            if self.state_diff:
                norm_cp_obs_mean = np.zeros((self.obs_space_dims*self.history_length,))
                norm_cp_obs_std = np.ones((self.obs_space_dims*self.history_length,))
            else:
                norm_cp_obs_mean = self.normalization['cp_obs']['mean']
                norm_cp_obs_std = self.normalization['cp_obs']['std']
            norm_cp_act_mean = self.normalization['cp_act']['mean']
            norm_cp_act_std = self.normalization['cp_act']['std']
        else:
            norm_obs_mean = np.zeros((self.obs_space_dims,))
            norm_obs_std = np.ones((self.obs_space_dims,))
            norm_act_mean = np.zeros((self.action_space_dims,))
            norm_act_std = np.ones((self.action_space_dims,))
            norm_delta_mean = np.zeros((self.obs_space_dims,))
            norm_delta_std = np.ones((self.obs_space_dims,))
            norm_cp_obs_mean = np.zeros((self.obs_space_dims*self.history_length,))
            norm_cp_obs_std = np.ones((self.obs_space_dims*self.history_length,))
            norm_cp_act_mean = np.zeros((self.action_space_dims*self.history_length,))
            norm_cp_act_std = np.ones((self.action_space_dims*self.history_length,))

        return (torch.FloatTensor(norm_obs_mean).to(self.device), torch.FloatTensor(norm_obs_std).to(self.device), torch.FloatTensor(norm_act_mean).to(self.device), torch.FloatTensor(norm_act_std).to(self.device), torch.FloatTensor(norm_delta_mean).to(self.device), torch.FloatTensor(norm_delta_std).to(self.device),
                torch.FloatTensor(norm_cp_obs_mean).to(self.device), torch.FloatTensor(norm_cp_obs_std).to(self.device), torch.FloatTensor(norm_cp_act_mean).to(self.device), torch.FloatTensor(norm_cp_act_std).to(self.device))

    def _preprocess_inputs(self, obs, act, delta, cp_obs, cp_act, future_bool, obs_next,  sim_param):   
        _sim_param = sim_param.reshape(-1, 1, 1)
        _sim_param = np.tile(sim_param, (1, 1, int(obs.shape[1]/self.obs_space_dims)))
        _sim_param = _sim_param.reshape((-1, 1))
        _future_bool= future_bool.reshape(-1)
        _obs = obs.reshape((-1, self.obs_space_dims))
        _act = act.reshape((-1, self.action_space_dims))
        _delta = delta.reshape((-1, self.obs_space_dims))
        _obs_next = obs_next.reshape((-1, self.obs_space_dims))

        _cp_obs = np.tile(cp_obs, (1, self.future_length))
        _cp_obs = _cp_obs.reshape((-1, self.obs_space_dims*self.history_length))
        _cp_act = np.tile(cp_act, (1, self.future_length))
        _cp_act = _cp_act.reshape((-1, self.action_space_dims*self.history_length))

        _obs = _obs[_future_bool>0, :]
        _act = _act[_future_bool>0, :]
        _delta = _delta[_future_bool>0, :]
        _obs_next = _obs_next[_future_bool>0, :]
        _cp_obs = _cp_obs[_future_bool>0, :]
        _cp_act = _cp_act[_future_bool>0, :]
        _sim_param = _sim_param[_future_bool > 0, :]
        return _obs, _act, _delta, _obs_next, _cp_obs, _cp_act, _sim_param

    def evaluate(self, shared_policy, shared_policy_reward, nested_policy_reward, n_episodes=10, writer=None, t=0):
        (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std) = self.get_normalization_stats()
        eval_transfer_rewards = [[0 for _ in self.data_types] for _ in self.data_types]
        for i, source in enumerate(self.data_types):
            for j, target in enumerate(self.data_types):
                if not (i==0 and j==1):
                    continue
                name = '%s-%s'%(source, target)
                item = self.nested_agents[i].sample_single()
                cp_obs, cp_act = item[3], item[4]
                if cp_obs is None:
                    continue
                source_context_tensor = self.cp(cp_obs, cp_act, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std).detach()
                source_context_tensor = torch.mean(source_context_tensor, dim=0, keepdim=True)
                reward = self.online_test(j, n_episodes=n_episodes, source_context_tensor=source_context_tensor, shared_policy=shared_policy[i])
                writer.add_scalar('%s/transfer_policy_reward'%name, reward.mean(), t+1)
                eval_transfer_rewards[i][j] = reward.mean()

        print('transfer reward', eval_transfer_rewards)

        return eval_transfer_rewards


    def online_test(self, idx, n_episodes, source_context_tensor=None, shared_policy=None):
        (norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std) = self.get_normalization_stats()
        
        reward_buffer = []
        for episode in (range(n_episodes)):
            self.env.set_data_type(self.data_types[idx])
            obs = self.env.reset()
            cp_obs = np.zeros((self.history_length, self.env.observation_space.shape[0]))
            cp_act = np.zeros((self.history_length, self.env.action_space.shape[0]))
            done = False
            episode_r = 0.
            while not done:
                obs_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
                cp_obs_tensor = torch.FloatTensor(cp_obs).reshape((-1, self.history_length*self.env.observation_space.shape[0])).to(self.device)
                cp_act_tensor = torch.FloatTensor(cp_act).reshape((-1, self.history_length*self.env.action_space.shape[0])).to(self.device)
                source_context = source_context_tensor.reshape(1, -1).to(self.device)
                target_context = self.cp(cp_obs_tensor, cp_act_tensor, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std).to(self.device)
                good_action =shared_policy.select_action(obs)
                good_action = torch.FloatTensor(good_action).reshape(1,-1).to(self.device)
                act = self.model(obs_tensor,good_action).cpu().data.numpy().flatten()
                new_obs, r, done, info = self.env.step(act)
                cp_obs[:-1] = cp_obs[1:]
                cp_act[:-1] = cp_act[1:]
                cp_act[-1] = act
                if self.state_diff:
                    cp_obs[-1] = copy.deepcopy(new_obs-obs)
                else:
                    cp_obs[-1] = copy.deepcopy(obs)
                obs = new_obs
                episode_r += r
            reward_buffer.append(episode_r)
        episode_r = sum(reward_buffer)
        # print('average reward: {:.2f}'.format(episode_r/episode_n))
        return np.array(reward_buffer)

def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)

def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean

def create_normalization(shape):
        return {'sum':np.zeros(shape, np.float32), 'square_sum':np.zeros(shape, np.float32), 'mean':np.zeros(shape, np.float32), 'std':np.ones(shape, np.float32), 'n':0}

