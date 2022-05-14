import torch
import time
from tabulate import tabulate
from statistics import mean


def load_pytorch_policy():
    """ Load a pytorch policy from saved file."""

    model = torch.load('models.pt')

    # make function for producing an action given a single state
    def get_action(o):
        with torch.no_grad():
            o = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in o.items()}
            action = model.act(torch.cat((o['observation'],o['desired_goal'])))
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    """ Run the policy during some episodes """

    logger = {
        'EpRet' : [],
        'EpLen': [],
    }

    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger['EpRet'].append(ep_ret)
            logger['EpLen'].append(ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    data = [['AverageEpRet', mean(logger['EpRet'])],
            ['AverageEpLen', mean(logger['EpLen'])]]
    print(tabulate(data))

    for key in logger:
        logger[key] = []
