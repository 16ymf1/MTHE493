import matplotlib.pyplot as plt
import numpy as np

folder = 'test'

ROWS = [190, 300, 324, 958]
skip = True

values = np.load(f'{folder}/reward_list.npy')
delivered = np.load(f'{folder}/delivered_list.npy')
time = np.load(f'{folder}/time_list.npy')
dist = np.load(f'{folder}/dist_list.npy')
total_dist = np.load(f'{folder}/total_dist.npy')
total_time = np.load(f'{folder}/total_avg_time.npy')
cells = np.load(f'{folder}/cell_tracker.npy', allow_pickle=True)

#plt.plot(values)
#plt.xlabel('Days Simulated')
#plt.ylabel('Reward')
#plt.show()

#plt.plot(delivered)
#plt.xlabel('Days Simulated')
#plt.ylabel('Orders Delivered')
#plt.show()

#plt.plot(time)
#plt.xlabel('Days Simulated')
#plt.ylabel('Average Order Time')
#plt.show()

#plt.plot(dist)
#plt.xlabel('Days Simulated')
#plt.ylabel('Average Order Dist')
#plt.show()

if not skip:
    plt.plot(total_time)
    plt.xlabel('Days Simulated')
    plt.ylabel('Average TOTAL Order Time')
    plt.show()

    plt.plot(total_dist)
    plt.xlabel('Days Simulated')
    plt.ylabel('Average TOTAL Order Dist')
    print(f'{total_dist[-1]}')
    plt.show()

for row in ROWS:
    for i, col in enumerate(cells[row]):
        plt.plot(cells[row][i], label=f'C{i+1}')
    plt.title(f'Cells for state {row}')
    plt.legend()
    plt.xlabel('Orders')
    plt.ylabel('Q Table Reward')
    plt.show()