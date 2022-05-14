import gym
from gym.wrappers import FilterObservation, FlattenObservation
from ddpg import ddpg
from her import her
import test


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchReach-v1')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    env = env = gym.make(args.env, reward_type='sparse')

    if (args.mode == 'train'):
        policy = ddpg(env, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, seed=args.seed)
        her(env, policy, epochs=args.epochs)

    elif (args.mode == 'test'):
        get_action = test.load_pytorch_policy()
        test.run_policy(env, get_action)
