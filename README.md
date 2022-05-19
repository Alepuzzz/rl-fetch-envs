This repository collects the code developed, the training results and the trained agents that are part of my Final Degree Project. This project is entitled "Reinforcement learning applied to the control of a robot manipulator" and was developed by me (Ángel Alepuz Jerez) under the supervision of my tutor Jorge Calvo Zaragoza during my last year of the Degree in Robotics Engineering at the University of Alicante.

## OBJECTIVE

The objective of this repository is to present different approaches to solve the robotic environments of the Fetch manipulator arm from the Gym library.

<p align="center">
  <img src="https://raw.githubusercontent.com/Alepuzzz/rl-fetch-envs/master/images/envs.png" width="250"/>
</p>

![envs](https://raw.githubusercontent.com/Alepuzzz/rl-fetch-envs/master/images/envs.png=x250)


## INSTALLATION

### Own implementation and Stable Baselines3 library

For working on my implementation (_ddpg_ and _ddpg\_her_ folders) and the codes that make use of Stable Baselines3 (_stable\_baselines3_ folder) it is necessary to install the following libraries:

- OpenAI Gym library (tested in version 0.19.0). 
```
pip install gym==0.19.0
```

- Stable Baselines3 (tested in version 1.4.0). [Installation instructions](https://github.com/DLR-RM/stable-baselines3).

- Mujoco (tested in version 2.1.2.14). [Installation instructions](https://github.com/openai/mujoco-py). 

It is recommended to use a virtual environment to avoid version conflicts between packages. In my case, Python 3.8.12 version has been used.

### Baselines library

In order to train and test the agents that use the Baselines library (_baselines_ folder) it is necessary to install the following libraries:

- OpenAI Gym library (tested in version 0.19.0). 
```
pip install gym==0.15.7
```

- Baselines (final version in maintenance status). [Installation instructions](https://github.com/openai/baselines).

- Mujoco (tested in version 2.1.2.14). [Installation instructions](https://github.com/openai/mujoco-py).

It is recommended to use another independent virtual environment to avoid conflicts between package versions. In my case, Python 3.6.13 version has been used.