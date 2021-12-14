import numpy as np
import torch
import torch.nn as nn
import gym
import argparse
import os
import utils
import TD3
from tensorboardX import SummaryWriter
import copy
from single_transfer import ActionTransferAgent
import logger
import sys
from gym.wrappers.time_limit import TimeLimit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_path(path):
	if not os.path.exists(path):
		os.mkdir(path)
	return path

def cat_env_name(envs_name):
	string = ''
	for env_name in envs_name[:-1]:
		string += env_name+'+'
	string += envs_name[-1]
	return string

# Runs policy for X episodes and returns average reward
def eval_policy(policy, cmodel, eval_env, data_types, history_length, eval_episodes, state_diff, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std):
	avg_reward = 0.
	avg_reward_task = [0. for _ in range(len(data_types))]
	num_episode_task = [0. for _ in range(len(data_types))]

	for _ in range(eval_episodes):
		data_type = np.random.choice(data_types)
		eval_env.set_data_type(data_type)
		state, eval_done = eval_env.reset(), False
		cp_obs = np.zeros((history_length, eval_env.observation_space.shape[0]))
		cp_act = np.zeros((history_length, eval_env.action_space.shape[0]))
		task_idx = data_types.index(data_type)
		num_episode_task[task_idx] += 1

		while not eval_done:
			cp_obs_tensor = torch.FloatTensor(cp_obs).reshape(1,-1).to(device)
			cp_act_tensor = torch.FloatTensor(cp_act).reshape(1,-1).to(device)
			context_tensor = cmodel(cp_obs_tensor, cp_act_tensor, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std)
			action = policy[task_idx].select_action(np.array(state))
			next_state, reward, eval_done, _ = eval_env.step(action)
			cp_obs[:-1] = cp_obs[1:]
			cp_act[:-1] = cp_act[1:]
			cp_act[-1] = copy.deepcopy(action)
			if state_diff:
				cp_obs[-1] = copy.deepcopy(next_state-state)
			else:
				cp_obs[-1] = copy.deepcopy(state)
			avg_reward += reward
			avg_reward_task[task_idx] += reward
			state = next_state

	avg_reward /= eval_episodes
	for i in range(len(data_types)):
		if num_episode_task[i] == 0:
			num_episode_task[i] = 1
			avg_reward_task[i] = 0.0
	avg_reward_task = np.array(avg_reward_task)/np.array(num_episode_task)
	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward, avg_reward_task


def eval_nested_policy(policy, eval_env, data_types, eval_episodes):
	nested_policy_reward = 0
	i = 0
	j = 1
	source = data_types[0]
	target = data_types[1]

	avg_reward = 0
	for _ in range(eval_episodes):
		eval_env.set_data_type(target)
		state, eval_done = eval_env.reset(), False
		while not eval_done:
			action = policy[i].select_action(np.array(state)) 
			next_state, reward, eval_done, _ = eval_env.step(action)
			avg_reward += reward
			state = next_state
	avg_reward /= eval_episodes
	nested_policy_reward = avg_reward
	return nested_policy_reward
				

def main(args):
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")
	if args.env == 'HalfCheetah-v2':
		data_types = ['arma0.1', 'arma0.5']
		from envs.halfcheetah_env import HalfCheetahEnv as Env
	elif args.env == 'Ant-v2':
		data_types = ['crip0', 'crip3']
		from envs.ant_cripple_env import AntEnv as Env
	data_type_str = cat_env_name(data_types)
	env = Env(data_types = data_types, seed=args.seed) 
	env = TimeLimit(env, max_episode_steps=env._max_episode_steps)

	log_path = args.trainLogDir
	model_path = args.model_path
	data_path = args.data_path
	writer = SummaryWriter(log_path)

	# Set seeds
	env.set_seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(args.seed)

	context_dim = args.context_dim
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	# Target policy smoothing is scaled wrt the action scale
	kwargs["policy_noise"] = args.policy_noise * max_action
	kwargs["noise_clip"] = args.noise_clip * max_action
	kwargs["policy_freq"] = args.policy_freq
	data_policy = [None for _ in data_types]
	policy = [None for _ in data_types]
	data_weight_paths = [args.data_weight_path_source, args.data_weight_path_target]
	good_weight_paths = [args.good_weight_path_source, args.good_weight_path_target]
	for i, data_type in enumerate(data_types):
		data_policy[i] = TD3.TD3(**kwargs)
		data_weight_path = data_weight_paths[i]
		good_weight_path = good_weight_paths[i]
		data_policy[i].actor.load_state_dict(torch.load(data_weight_path))
		policy[i] = TD3.TD3(**kwargs)
		policy[i].actor.load_state_dict(torch.load(good_weight_path))		

	action_transfer_agent = ActionTransferAgent(env_name=args.env, env=env, data_types=data_types, seed=args.seed, history_length=args.history_length, future_length=args.future_length, context_out_dim=args.context_dim, batch_size=args.batch_size, forward_n_update=args.forward_n_update, cycle_n_update=args.cycle_n_update, state_diff=args.state_diff, constractive_weight=args.constractive_weight, forward_lr=args.forward_lr)
	nested_policy_reward = 0.0
	nested_policy_reward = eval_nested_policy(policy, env, data_types, args.eval_n)

	task_idx = 1
	env.set_data_type(data_types[task_idx])
	state, done = env.reset(), False
	sim_param = env.get_sim_parameters()
	cp_obs = np.zeros((args.history_length, env.observation_space.shape[0]))
	cp_act = np.zeros((args.history_length, env.action_space.shape[0]))
	
	episode_reward = 0
	episode_timesteps = 0
	timesteps_since_eval = 0
	episode_num = 0

	# Prepare data
	if args.newData:
		print("Get data")
		for task_idx in [0,1]:
			our_now_curr, our_action_curr, our_next_curr = [], [], []
			done_curr, sim_param_curr, cp_obs_curr, cp_act_curr = [], [], [], []
			for t in range(int(10e4)):
				our_now_curr.append(state)
				action = data_policy[task_idx].select_action(np.array(state))
				our_action_curr.append(action)
				next_state, reward, done, _ = env.step(action)
				our_next_curr.append(next_state)
				done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

				done_curr.append(done)
				sim_param_curr.append(sim_param)
				cp_obs_curr.append(cp_obs)
				cp_act_curr.append(cp_act)
				action_transfer_agent.add(state, action, done, sim_param, cp_obs, cp_act)
				
				cp_obs[:-1] = cp_obs[1:]
				cp_act[:-1] = cp_act[1:]
				cp_act[-1] = copy.deepcopy(action)
				if args.state_diff:
					cp_obs[-1] = copy.deepcopy(next_state-state)
				else:
					cp_obs[-1] = copy.deepcopy(state)
				state = next_state
				episode_reward += reward
				episode_timesteps += 1
				if done:
					# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
					writer.add_scalar('train_reward/%s'%data_types[task_idx], episode_reward, t+1)
					print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Target: {data_types[task_idx]}")
					timesteps_since_eval += episode_timesteps
					episode_reward = 0
					episode_timesteps = 0
					episode_num += 1

					# Reset environment
					env.set_data_type(data_types[task_idx])
					state, done = env.reset(), False
					sim_param = env.get_sim_parameters()
					cp_obs = np.zeros((args.history_length, env.observation_space.shape[0]))
					cp_act = np.zeros((args.history_length, env.action_space.shape[0]))

			np.save(os.path.join(data_path, '%s-action-%d.npy'%(args.env, task_idx)), our_action_curr)
			np.save(os.path.join(data_path, '%s-now-%d.npy'%(args.env, task_idx)), our_now_curr)
			np.save(os.path.join(data_path, '%s-done-%d.npy'%(args.env, task_idx)), done_curr)
			np.save(os.path.join(data_path, '%s-sim-%d.npy'%(args.env, task_idx)), sim_param_curr)
			np.save(os.path.join(data_path, '%s-cpobs-%d.npy'%(args.env, task_idx)), cp_obs_curr)
			np.save(os.path.join(data_path, '%s-cpact-%d.npy'%(args.env, task_idx)), cp_act_curr)
	else:
		print("Load data")
		for task_idx in [0,1]:
			our_action_curr    = np.load(safe_path(os.path.join(data_path, '%s-action-%d.npy'%(args.env, task_idx))))
			our_now_curr       = np.load(safe_path(os.path.join(data_path, '%s-now-%d.npy'%(args.env, task_idx))))
			our_done_curr      = np.load(safe_path(os.path.join(data_path, '%s-done-%d.npy'%(args.env, task_idx))))
			our_sim_param_curr = np.load(safe_path(os.path.join(data_path, '%s-sim-%d.npy'%(args.env, task_idx))))
			our_cp_obs_curr    = np.load(safe_path(os.path.join(data_path, '%s-cpobs-%d.npy'%(args.env, task_idx))))
			our_cp_act_curr    = np.load(safe_path(os.path.join(data_path, '%s-cpact-%d.npy'%(args.env, task_idx))))

			for t in range(int(10e4)):
				action_transfer_agent.add(our_now_curr[t], our_action_curr[t], our_done_curr[t], our_sim_param_curr[t], our_cp_obs_curr[t], our_cp_act_curr[t])


	# Prepare forward model
	if not args.newForward:
		print("Load forward model")
		action_transfer_agent.mlp.load_state_dict(torch.load(safe_path(os.path.join(model_path, '%s-mlp'%(args.env)))))
		action_transfer_agent.cp.load_state_dict(torch.load(safe_path(os.path.join(model_path, '%s-cp'%(args.env)))))
	else:
		print("Train Forward model")
		for t in range(30):
			action_transfer_agent.train_context_forward(writer=writer, t=t)
		torch.save(action_transfer_agent.mlp.state_dict(), safe_path(os.path.join(model_path, '%s-mlp'%(args.env))))
		torch.save(action_transfer_agent.cp.state_dict(), safe_path(os.path.join(model_path, '%s-mlp'%(args.env))))
	

	# Prepare transfer model
	if args.newTransfer:
		curr_best_eval = -1
		print("Train transfer model and eval")
		for t in range(30):	
			action_transfer_agent.train_transfer_agent_bisimilar(writer=writer, t=t)		
			eval_transfer_reward = action_transfer_agent.evaluate(shared_policy=policy, shared_policy_reward=avg_reward_task, nested_policy_reward=nested_policy_reward, n_episodes=args.eval_n, writer=writer, t=t)
			logger.logkv("eval_transfer_reward", eval_transfer_reward[0][1])
			logger.logkv("train_epoch", t)
			logger.dumpkvs()
	else:
		action_transfer_agent.model.load_state_dict(torch.load(safe_path(os.path.join(model_path, '%s-model'%(args.env)))))
		(norm_obs_mean, norm_obs_std, norm_act_mean, norm_act_std, norm_delta_mean, norm_delta_std, norm_cp_obs_mean, norm_cp_obs_std, norm_cp_act_mean, norm_cp_act_std) = action_transfer_agent.get_normalization_stats()
		action_transfer_agent.model.eval()
		avg_reward, avg_reward_task = eval_policy(policy=policy, cmodel=action_transfer_agent.cp, eval_env=env, eval_episodes=len(data_types), history_length=args.history_length, data_types=data_types, state_diff=args.state_diff, norm_cp_obs_mean=norm_cp_obs_mean, norm_cp_obs_std=norm_cp_obs_std, norm_cp_act_mean=norm_cp_act_mean, norm_cp_act_std=norm_cp_act_std)
		eval_transfer_reward = action_transfer_agent.evaluate(shared_policy=policy, shared_policy_reward=avg_reward_task, nested_policy_reward=nested_policy_reward, n_episodes=args.eval_n, writer=writer, t=t)
		logger.logkv("eval_transfer_reward", eval_transfer_reward[0][1])
		logger.logkv("train_epoch", t)
		logger.dumpkvs()

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default='TD3', type=str)
	parser.add_argument("--env", default='no_specify', type=str)
	parser.add_argument("--reward_delay", default=1, type=int) 
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=10e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)               # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	# parser.add_argument("--log_root", default="results_transfer_action/")
	parser.add_argument("--history_length", default=10, type=int)
	parser.add_argument("--future_length", default=10, type=int)
	parser.add_argument("--context_dim", default=10, type=int)
	parser.add_argument("--forward_n_update", default=10000, type=int)
	parser.add_argument("--cycle_n_update", default=3000, type=int)
	parser.add_argument("--eval_n", default=100, type=int)
	parser.add_argument("--state_diff", default=True, type=bool)
	parser.add_argument("--constractive_weight", default=1, type=float)
	parser.add_argument("--forward_lr", default=2e-4, type=float)

	# log
	parser.add_argument("--logDir", default="results_transfer_action", type=str)
	parser.add_argument("--trainLogDir", default="results_transfer_action", type=str)
	# policy
	parser.add_argument("--data_weight_path_source", default="policy/source_data_policy", type=str)
	parser.add_argument("--data_weight_path_target", default="policy/target_data_policy", type=str)
	parser.add_argument("--good_weight_path_target", default="policy/target_good_policy", type=str)
	parser.add_argument("--good_weight_path_source", default="policy/source_good_policy", type=str)
	# data
	parser.add_argument("--data_path", default="data/", type=str)
	# model
	parser.add_argument("--model_path", default="model/", type=str)


	parser.add_argument("--newData", default=False, type=bool)
	parser.add_argument("--newForward", default=False, type=bool)
	parser.add_argument("--newTransfer", default=False, type=bool)

	args = parser.parse_args()

	logger.configure(args.logDir)
	main(args)
