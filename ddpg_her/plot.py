import pandas
import numpy as np
import matplotlib.pyplot as plt

experiment_name = 'slide'

# Read data
df = pandas.read_csv('data/' + experiment_name + '_logs.csv')

# Show and save figure
plt.figure()
plt.plot(df['Epoch'],df['TestEpRew'])
plt.title('Recompensa media de los episodios de test')
plt.xlabel('Época (Cada época = 80 episodios = 80x50 timesteps)')
plt.ylabel('Recompensa media')
plt.savefig('results/' + experiment_name + '_ep_rew_mean.png')
plt.show()

# Show and save figure
plt.figure()
plt.plot(df['Epoch'],df['TestSuccessRate'])
plt.title('Tasa de éxito de los episodios de test')
plt.xlabel('Época (Cada época = 80 episodios = 80x50 timesteps)')
plt.ylabel('Tasa de éxito')
plt.savefig('results/' + experiment_name + '_success_rate.png')
plt.show()
