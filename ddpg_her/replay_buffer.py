import numpy as np
from sympy import O
import torch

class HerReplayBuffer:
    """
    A simple FIFO experience replay buffer for goal based environments.
    """

    def __init__(self, obs_dim, act_dim, goal_dim, size, n_sampled_goal=4):

        self.obs_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self._combined_shape(size, act_dim), dtype=np.float32)
        self.d_goal_buf = np.zeros(self._combined_shape(size, goal_dim), dtype=np.float32)
        self.a_goal_buf = np.zeros(self._combined_shape(size, goal_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.transition_info = []
        self.episode_start = 0
        self.n_sampled_goal = n_sampled_goal
        self.ptr, self.size, self.max_size = 0, 0, size
        


    def store(self, obs, act, rew, next_obs, done, info):
        self.obs_buf[self.ptr] = obs['observation']
        self.obs2_buf[self.ptr] = next_obs['observation']
        self.act_buf[self.ptr] = act
        self.d_goal_buf[self.ptr] = next_obs['desired_goal']
        self.a_goal_buf[self.ptr] = next_obs['achieved_goal']
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.transition_info.append(info)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     goal=self.d_goal_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


    def sample_additional_goals(self, env, k=4):

        episode_end = self.ptr

        if self.episode_start < episode_end:
            idxs = np.arange(self.episode_start,episode_end)
        else:
            idxs = np.append(np.arange(self.episode_start,self.size),
                             np.arange(0,episode_end))

        
        # Iterate over the episode excluding the last transition
        for idx, t in enumerate(idxs[:-1]): 
            
            goals_idxs = np.random.choice(idxs[idx+1:],k)

            for g_idx in goals_idxs:
                o = dict(observation=self.obs_buf[t])
                o2 = dict(observation=self.obs2_buf[t],
                          achieved_goal=self.a_goal_buf[t],
                          desired_goal=self.a_goal_buf[g_idx]) # Substitute desired goal
                a = self.act_buf[t]
                info = self.transition_info[idx]
                r = env.compute_reward(o2['achieved_goal'], o2['desired_goal'], info)
                d = False # TODO: ALWAYS?
                self.store(o, a, r, o2, d, info)

        """
        for idx, t in enumerate(idxs): 
            o = dict(observation=self.obs_buf[t],
                     achieved_goal=self.a_goal_buf[t],
                     desired_goal=self.a_goal_buf[episode_end]) # Substitute desired goal)
            o2 = dict(observation=self.obs2_buf[t])
            a = self.act_buf[t]
            info = self.transition_info[idx]
            r = env.compute_reward(o['achieved_goal'], o['desired_goal'], info)
            d = self.done_buf[t] if (idx!=idxs[-1]) else True
            self.store(o, a, r, o2, d, info)
        """

        self.transition_info = []
        self.episode_start = self.ptr


    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)
