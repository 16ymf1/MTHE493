from Environment import SimulationEnvironment
from QLearningModel import QLearningModel
import numpy as np

ORDER_RATE = 0.09
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.95
NUM_EPISODES = 5000
NUM_TIMESTEPS = 288

folder = 'Results'

q_table = f'{folder}/q_table.npy'


restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]
sim = SimulationEnvironment(10, restuarants, couriers, bin_size=14.267)
model = QLearningModel(LEARNING_RATE, DISCOUNT_RATE, NUM_EPISODES, NUM_TIMESTEPS, sim, ORDER_RATE, q_table)
model.run_sim()

np.save('Updated_Results/reward_list.npy', model.reward_list)
np.save('Updated_Results/delivered_list.npy', model.order_delivered_list)
np.save('Updated_Results/time_list.npy', model.order_time_list)
np.save('Updated_Results/dist_list.npy', model.order_dist_list)
np.save('Updated_Results/total_dist.npy', model.total_avg_dist)
np.save('Updated_Results/total_avg_time.npy', model.total_avg_time)
np.save('Updated_Results/q_table.npy', model.Q)
np.savetxt('Updated_Results/q_table.csv', model.Q, delimiter=',')
np.savetxt('Updated_Results/q_table_tracker.csv', model.Q_tracker, delimiter=',')