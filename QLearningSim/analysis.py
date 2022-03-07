import matplotlib.pyplot as plt
import numpy as np

folder = 'Results'

values = np.load(f'{folder}/reward_list.npy')
delivered = np.load(f'{folder}/delivered_list.npy')
time = np.load(f'{folder}/time_list.npy')
dist = np.load(f'{folder}/dist_list.npy')
total_dist = np.load(f'{folder}/total_dist.npy')
total_time = np.load(f'{folder}/total_avg_time.npy')

plt.plot(values)
plt.xlabel('Days Simulated')
plt.ylabel('Reward')
plt.show()

plt.plot(delivered)
plt.xlabel('Days Simulated')
plt.ylabel('Orders Delivered')
plt.show()

plt.plot(time)
plt.xlabel('Days Simulated')
plt.ylabel('Average Order Time')
plt.show()

plt.plot(dist)
plt.xlabel('Days Simulated')
plt.ylabel('Average Order Dist')
plt.show()

plt.plot(total_time)
plt.xlabel('Days Simulated')
plt.ylabel('Average TOTAL Order Time')
plt.show()

plt.plot(total_dist)
plt.xlabel('Days Simulated')
plt.ylabel('Average TOTAL Order Dist')
plt.show()