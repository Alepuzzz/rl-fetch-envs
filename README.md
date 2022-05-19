Este repositorio recoge el código desarrollado, los resultados del entrenamiento y los agentes entrenados que se reflejan en mi Trabajo Fin de Grado (TFG). Este proyecto se titula "Aprendizaje por refuerzo aplicado al control de un robot manipulador" y fue desarrollado por mi (Ángel Alepuz Jerez) bajo la supervisión de mi tutor Jorge Calvo Zaragoza durante mi último año del Grado en Ingeniería Robótica en la Universidad de Alicante (UA).

# OBJETIVO

El objetivo de este repositorio es presentar distintas aproximaciones para solucionar los entornos robóticos del brazo manipulador Fetch de la librería de Gym.

![envs](https://raw.githubusercontent.com/Alepuzzz/rl-fetch-envs/master/images/envs.png)


## INSTALACIÓN

### Implementación propia y librería Stable Baselines3

Para el correcto funcionamiento de mi implementación (carpetas _ddpg_ y _ddpg\_her_) y de los códigos que hacen uso de Stable Baselines3 (carpeta _stable\_baselines3_) es necesario instalar las siguientes librerías:

- Stable Baselines3 (probado en su versión 1.4.0). Las intrucciones de [instalación](https://github.com/DLR-RM/stable-baselines3) se encuentran en su repositorio oficial.

- Mujoco (probado en su versión 2.1.2.14). Las intrucciones de [instalación](https://github.com/openai/mujoco-py) se encuentran en su repositorio oficial.

Se recomienda emplear un entorno virtual para evitar conflictos de versiones entre paquetes. En mi caso se ha utilizado Python 3.8.12.

### Librería Baselines

Para poder entrenar y probar los agentes entrados usando la librería Baselines (carpeta _baselines_) es necesario instalar las siguientes librerías:

- Baselines (versión final en estado de mantenimiento). Las intrucciones de [instalación](https://github.com/openai/baselines) se encuentran en su repositorio oficial.

- Mujoco (probado en su versión 2.1.2.14). Las intrucciones de [instalación](https://github.com/openai/mujoco-py) se encuentran en su repositorio oficial.

Se recomienda emplear otro entorno virtual independiente para evitar conflitos entre versiones de paquetes. En mi caso se ha utilizado Python 3.6.13.


