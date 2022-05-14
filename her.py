import gym
import numpy as np

from stable_baselines3 import HerReplayBuffer, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

env = gym.make('FetchReach-v1')
"""
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize the model
model = DDPG(
    "MultiInputPolicy", 
    env, 
    action_noise=action_noise,
    replay_buffer_class=HerReplayBuffer, 
    verbose=1,
)

# Train the model
model.learn(20000) # 8 min aprox

model.save("./her")
"""
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = DDPG.load('./her', env=env)

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

