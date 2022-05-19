import gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make('FetchReach-v1') #, reward_type='dense')


# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize the model
model = DDPG(
    "MultiInputPolicy", 
    env, 
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./ddpg_sparse_tensorboard/"
)

# Train the model
model.learn(100000)

model.save("./ddpg")

#model = DDPG.load('./ddpg')

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

