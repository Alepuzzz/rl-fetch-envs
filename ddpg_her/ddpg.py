from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
from tabulate import tabulate
from statistics import mean
import pandas as pd

import models


class ddpg:

    def __init__(self, env, model_path='trained_agent.pt', logs_path='logs.csv', actor_critic=models.MLPActorCritic, 
            ac_kwargs=dict(), seed=0, gamma=0.98, polyak=0.95, pi_lr=1e-3, q_lr=1e-3, max_ep_len=50):
        """
        Deep Deterministic Policy Gradient (DDPG)


        Args:
            env : An environment that must satisfy the OpenAI Gym API.

            model_path: Path for saving the trained agent.

            logs_path: Path for saving the csv containing the training logs.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, and a ``q`` module. The ``act`` method and
                ``pi`` module should accept batches of observations as inputs,
                and ``q`` should accept a batch of observations and a batch of 
                actions as inputs. When called, these should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``q``        (batch,)          | Tensor containing the current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to DDPG.

            seed (int): Seed for random number generators.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. (Always between 0 and 1, usually close to 1.)

            pi_lr (float): Learning rate for policy.

            q_lr (float): Learning rate for Q-networks.
        """
        self.model_path = model_path
        self.logs_path = logs_path
        self.start_time = time.time()
        self.gamma = gamma
        self.polyak = polyak
        self.max_ep_len = max_ep_len

        self.logger = {
            'EpRet' : [],
            'TestEpRet' : [],
            'EpLen': [],
            'TestEpLen' : [],
            'TestSuccessRate': [],
            'LossPi' : [],
            'LossQ' : []
        }

        self.csv_logger = {
            'Epoch' : [],
            'TestEpRew' : [],
            'TestSuccessRate': []
        }

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.test_env = deepcopy(env)

        self.act_dim = env.action_space.shape[0]
        # Action limit for clamping
        #self.act_limit = env.action_space.high[0]
        self.act_limit = 5

        # Create actor-critic module and target networks
        self.ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Count variables 
        var_counts = tuple(models.count_vars(module) for module in [self.ac.pi, self.ac.q])
        print('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)


    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = torch.cat((data['obs'], data['goal']),1), data['act'], data['rew'], torch.cat((data['obs2'], data['goal']),1), data['done']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        return loss_q


    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = torch.cat((data['obs'], data['goal']),1)
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()


    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        self.logger['LossQ'].append(loss_q.item())
        self.logger['LossPi'].append(loss_pi.item())

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def get_action(self, o, noise_scale):
        o = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in o.items()}
        a = self.ac.act(torch.cat((o['observation'],o['desired_goal'])))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


    def test_agent(self, num_test_episodes):
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not(d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, info = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1

            self.logger['TestEpRet'].append(ep_ret)
            self.logger['TestEpLen'].append(ep_len)
            self.logger['TestSuccessRate'].append(info['is_success'])
    

    def print_epoch_data(self, epoch, t, episode_count):
        
        epochData = [['Epoch', epoch],
                    ['EpisodeCount', episode_count],
                    ['AverageEpRet', mean(self.logger['EpRet'])],
                    ['AverageTestEpRet', mean(self.logger['TestEpRet'])],
                    ['AverageEpLen', mean(self.logger['EpLen'])],
                    ['TestEpLen', mean(self.logger['TestEpLen'])],
                    ['TestSucessRate', mean(self.logger['TestSuccessRate'])],
                    ['TotalEnvInteracts', t],
                    ['LossPi', mean(self.logger['LossPi'])],
                    ['LossQ', mean(self.logger['LossQ'])],
                    ['Time', time.time()-self.start_time]]
        print(tabulate(epochData))

        self.csv_logger['Epoch'].append(epoch)
        self.csv_logger['TestEpRew'].append(mean(self.logger['TestEpRet']))
        self.csv_logger['TestSuccessRate'].append(mean(self.logger['TestSuccessRate']))

        for key in self.logger:
            self.logger[key] = []
    

    def save_model(self):
        torch.save(self.ac, self.model_path)

    def save_logs(self):
        df =  pd.DataFrame(self.csv_logger)
        df.to_csv(self.logs_path)
