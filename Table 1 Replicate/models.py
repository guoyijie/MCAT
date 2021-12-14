import torch
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean

class SiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class ContextModel(nn.Module):
    def __init__(self, context_dim, context_hidden_sizes=(256, 128, 64), context_out_dim=10, context_weight_decays=(0.000025, 0.00005, 0.000075), context_hidden_nonlinearity='relu', device='cpu'):
        
        super(ContextModel,self).__init__()
        self.context_weight_decays = context_weight_decays
        context_hidden_sizes = [context_dim]+list(context_hidden_sizes)
        self.context_hidden_sizes = context_hidden_sizes

        if context_hidden_nonlinearity=='relu':
            activation_fn = nn.ReLU()
        elif hidden_nonlinearity=='swish':
            activation_fn = SiLU()

        self.layers = nn.Sequential()
        
        for idx in range(len(context_hidden_sizes) - 1):
            self.layers.add_module('linear_%02d'%idx, nn.Linear(context_hidden_sizes[idx], context_hidden_sizes[idx+1]))
            self.layers.add_module('activation_%02d'%idx, activation_fn)
        self.layers.add_module('linear_-1', nn.Linear(context_hidden_sizes[-1], context_out_dim))
        self.layers = self.layers.to(device)
        self.device = device

    def l2_reg(self):
        l2_regs = torch.tensor(0.).to(self.device)
        for p in self.layers.named_parameters():
            idx = int(float(p[0][7:9]))
            l2_regs += torch.norm(p[-1])*self.context_weight_decays[idx]
        return l2_regs

    def forward(self, cp_obs, cp_act, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std):
        normalized_cp_obs = normalize(cp_obs, norm_cp_obs_mean, norm_cp_obs_std)
        normalized_cp_act = normalize(cp_act, norm_cp_act_mean, norm_cp_act_std)
        normalized_cp_x = torch.cat([normalized_cp_obs, normalized_cp_act], dim=-1)
        cp_output = self.layers(normalized_cp_x)
        return cp_output

class ContextForwardModel(nn.Module):
    def __init__(self, input_obs_dim, input_act_dim, output_dim, context_out_dim=10, hidden_sizes=(200,200,200,200), weight_decays=(0.000025, 0.00005, 0.000075, 0.000075, 0.0001), hidden_nonlinearity='swish', device='cpu'): 
        super(ContextForwardModel,self).__init__()
        self.weight_decays = weight_decays
        hidden_sizes = [input_obs_dim+input_act_dim+context_out_dim] + list(hidden_sizes)
        if hidden_nonlinearity=='relu':
            activation_fn = nn.ReLU()
        elif hidden_nonlinearity=='swish':
            activation_fn = SiLU()

        self.layers = nn.Sequential()

        for idx in range(len(hidden_sizes) - 1):
            self.layers.add_module('linear_%02d'%idx, nn.Linear(hidden_sizes[idx], hidden_sizes[idx+1]))
            self.layers.add_module('activation_%02d'%idx, activation_fn)

        self.layers = self.layers.to(device)
        self.mu_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], output_dim)).to(device)
        self.logvar_layer = nn.Sequential(nn.Linear(hidden_sizes[-1], output_dim)).to(device)

        self.max_logvar = torch.FloatTensor(np.ones([1, output_dim])/2.).to(device)
        self.min_logvar = torch.FloatTensor(-np.ones([1, output_dim])*10).to(device)
        self.device = device

    def l2_reg(self):
        l2_regs = torch.tensor(0.).to(self.device)
        for p in self.layers.named_parameters():
            idx = int(float(p[0][7:9]))
            l2_regs += torch.norm(p[-1])*self.weight_decays[idx]
        for p in self.mu_layer.parameters():
            l2_regs += torch.norm(p)*self.weight_decays[-1]
        for p in self.logvar_layer.parameters():
            l2_regs += torch.norm(p)*self.weight_decays[-1]
        return l2_regs

    def forward(self, obs, act, cp, norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, deterministic):
        normalized_obs = normalize(obs, norm_obs_mean, norm_obs_std)
        normalized_act = normalize(act, norm_act_mean, norm_act_std)

        x = torch.cat([normalized_obs, normalized_act, cp], 1)

        xx = self.layers(x)
        mu = self.mu_layer(xx)
        logvar = self.logvar_layer(xx)

        denormalized_mu = denormalize(mu, norm_delta_mean, norm_delta_std)

        if deterministic:
            xx = denormalized_mu
        else:
            logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)

            denormalized_logvar = logvar + 2 * torch.log(norm_delta_std)
            denormalized_std = torch.exp(denormalized_logvar / 2.0)
            xx = denormalized_mu + torch.randn(denormalized_mu.shape).to(self.device) * denormalized_std
        return xx, mu, logvar


class MultiPairAxmodel(nn.Module):
    def __init__(self,state_dim, action_dim, context_dim):
        super(MultiPairAxmodel,self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )
        self.action_fc = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.ReLU()
        )
        self.source_context_fc = nn.Sequential(
            nn.Linear(self.context_dim, 128),
            nn.ReLU()
        )
        self.target_context_fc = nn.Sequential(
            nn.Linear(self.context_dim, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, state, action, source_context, target_context):
        action = self.action_fc(action)
        state = self.state_fc(state)
        source_context = self.source_context_fc(source_context)
        target_context = self.target_context_fc(target_context)
        state = torch.cat((action, state, source_context, target_context), 1)
        action = self.fc(state)*self.max_action
        return action

class PairAxmodel(nn.Module):
    def __init__(self,state_dim, action_dim):
        super(PairAxmodel,self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU()
        )
        self.action_fc = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, state, action):
        action = self.action_fc(action)
        state = self.state_fc(state)
        state = torch.cat((action, state), 1)
        action = self.fc(state)*self.max_action
        return action

"""
class MultiPairAxmodel(nn.Module):
    def __init__(self,state_dim, action_dim, context_dim):
        super(MultiPairAxmodel,self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim+self.action_dim+self.context_dim*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, state, action, source_context, target_context):
        state = torch.cat((action, state, source_context, target_context), 1)
        action = self.fc(state)*self.max_action
        return action


class MultiPairAxmodel(nn.Module):
    def __init__(self,state_dim, action_dim, context_dim):
        super(MultiPairAxmodel,self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.fc = nn.Sequential(
            nn.Linear(self.state_dim+self.action_dim+self.context_dim*2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )
        self.max_action = 1.0

    def forward(self, state, action, source_context, target_context):
        state = torch.cat((action, state, source_context, target_context), 1)
        action = self.fc(state)*self.max_action
        return action
"""

class ConditionDmodel(nn.Module):
    def __init__(self,state_dim, action_dim, context_dim):
        super(ConditionDmodel,self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.state_fc = nn.Sequential(
            nn.Linear(self.action_dim, 256),
            nn.ReLU()
        )
        self.condition_fc = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, condition):
        s = self.state_fc(state)
        c = self.condition_fc(condition)
        return self.fc(torch.cat((s,c), dim=1))

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.use_lsgan = use_lsgan

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        if not self.use_lsgan:
            input = torch.nn.Sigmoid()(input)
        return self.loss(input, target_tensor)

class ImagePool():
    def __init__(self, pool_size, seed):
        self.pool_size = pool_size
        self.rng = np.random.RandomState(seed) 
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = self.rng.uniform(0, 1)
                if p > 0.5:
                    random_id = self.rng.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images

