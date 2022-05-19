import pandas
import matplotlib.pyplot as plt

experiment_name = 'her_FetchSlide-v1'

# Read data
df = pandas.read_csv('data/' + experiment_name + '_ep_rew_mean.csv')

# Show and save figure
plt.figure()
plt.plot(df['Step'],df['Value'])
plt.title('Recompensa media de los episodios')
plt.xlabel('Pasos')
plt.ylabel('Recompensa media')
plt.savefig('results/' + experiment_name + '_ep_rew_mean.png')
plt.show()

# Read data
df = pandas.read_csv('data/' + experiment_name + '_success_rate.csv')

# Show and save figure
plt.figure()
plt.plot(df['Step'],df['Value'])
plt.title('Tasa de éxito')
plt.xlabel('Pasos')
plt.ylabel('Tasa de éxito')
plt.savefig('results/' + experiment_name + '_success_rate.png')
plt.show()
