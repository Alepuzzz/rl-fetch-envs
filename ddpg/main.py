import gym
from gym.wrappers import FilterObservation, FlattenObservation
import argparse

from ddpg import ddpg
import test


def create_env(env_name):

    env = gym.make(env_name, reward_type='dense') # Reward type can be 'sparse' or 'dense'

    # Convert a goal based environment into a standard environment
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

    return env


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchReach-v1')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='trained_agent.pt')
    parser.add_argument('--logs_path', type=str, default='logs.csv')
    args = parser.parse_args()

    env = create_env(args.env)

    if (args.mode == 'train'):
        ddpg(env, model_path=args.model_path, logs_path=args.logs_path, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs, polyak=0.95, batch_size=256, act_noise=0.2)

    elif (args.mode == 'test'):
        get_action = test.load_pytorch_policy(path=args.model_path)
        test.run_policy(env, get_action)
