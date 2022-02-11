import matplotlib.pyplot as plt
import numpy as np

values = np.load('reward_list.npy')
delivered = np.load('delivered_list.npy')
plt.plot(values)
plt.xlabel('Days Simulated')
plt.ylabel('Reward')
plt.show()

plt.plot(delivered)
plt.xlabel('Days Simulated')
plt.ylabel('Orders Delivered')
plt.show()