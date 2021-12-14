import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
from tensorboardX import SummaryWriter

def make_env(env_name, data_type):
        env = gym.make(env_name)
        if env_name=='Hopper-v2':
            import sys
            sys.path.insert(0,'..')
            from envs.hopper_env import HopperEnv
            print(data_type)
            env = HopperEnv(data_types = [data_type], seed=args.seed)
        if env_name=='Walker2d-v2':
            import sys
            sys.path.insert(0,'..')
            from envs.walker2d_env import Walker2dEnv
            print(data_type)
            env = Walker2dEnv(data_types = [data_type], seed=args.seed) 
        if  env_name=='HalfCheetah-v2':
            import sys
            sys.path.insert(0,'..')
            from envs.halfcheetah_env import HalfCheetahEnv
            #from envs.halfcheetah_cripple_env import HalfCheetahEnv
            env = HalfCheetahEnv(data_types = [data_type], seed=args.seed)
        if  env_name=='Ant-v2':
            import sys
            sys.path.insert(0,'..')
            from envs.ant_cripple_env import AntEnv
            env = AntEnv(data_types = [data_type], seed=args.seed)
        if  env_name=='Humanoid-v2':
            import sys
            sys.path.insert(0,'..')
            from envs.humanoid_mass_env import HumanoidEnv
            env = HumanoidEnv(data_types = [data_type], seed=args.seed)
        from envs.normalized_env import NormalizedEnv
        from envs.delay_reward_env import DelayRewardEnv
        from gym.wrappers.time_limit import TimeLimit	
        env = NormalizedEnv(env)
        env = TimeLimit(env, max_episode_steps=env._max_episode_steps)
        env = DelayRewardEnv(env, reward_delay=1)
        env.set_seed(args.seed)
        env.action_space.seed(args.seed)
        return env



def safe_path(path):
	if not os.path.exists(path):
		os.mkdir(path)
	return path

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, data_type, eval_episodes=10):
	eval_env = make_env(env_name, data_type)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		episode_steps = 0
		while (not done) and episode_steps<eval_env._max_episode_steps:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward
			episode_steps +=1

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def main(args):
	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")
	log_path = safe_path(os.path.join(args.log_root, '{}_{}'.format(args.env, args.data_type)))
	result_path = safe_path(os.path.join(log_path, 'results'))
	model_path = safe_path(os.path.join(log_path, 'models'))
	writer = SummaryWriter(log_path)

	env = make_env(args.env, args.data_type)

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

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
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, args.data_type)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
					policy.select_action(np.array(state))
					+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done or episode_timesteps>=env._max_episode_steps:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(
				f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed, args.data_type))
			np.save(os.path.join(result_path, '{}'.format(file_name)), evaluations)
			writer.add_scalar("eval_reward", evaluations[-1], t+1)
		if (t + 1) % 2e4 == 0:
			if args.save_model: policy.save(os.path.join(model_path, '{}_epoch{}'.format(file_name, t+1)))


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="Hopper-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=4.1e5, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", default=True)               # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

	parser.add_argument("--data_type", default="size0.06")
	parser.add_argument("--log_root", default="logs/")
	args = parser.parse_args()

	main(args)
