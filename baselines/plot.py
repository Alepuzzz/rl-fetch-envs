import pandas
import numpy as np
import matplotlib.pyplot as plt

experiment_name = 'her_FetchSlide-v1'

# Read data
df = pandas.read_csv('data/' + experiment_name + '_test_success_rate.csv')

# Show and save figure
plt.figure(figsize=[12.8,4.8])
plt.plot(df['Step'],df['Value'])
plt.title('Tasa de éxito en los episodios de test')
plt.xlabel('Época (Cada época = 25 episodios x 4 entornos en paralelo = 25x4x50 timesteps)')
plt.ylabel('Tasa de éxito')
plt.savefig('results/' + experiment_name + '_test_success_rate.png')
plt.show()
