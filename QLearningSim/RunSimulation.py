from Environment import SimulationEnvironment
from QLearningModel import QLearningModel
import numpy as np

ORDER_RATE = 0.08
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.9
NUM_EPISODES = 40000
NUM_TIMESTEPS = 288

save_folder = ''


restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]
sim = SimulationEnvironment(10, restuarants, couriers, bin_size=14.267)
model = QLearningModel(LEARNING_RATE, DISCOUNT_RATE, NUM_EPISODES, NUM_TIMESTEPS, sim, ORDER_RATE)
model.run_sim()

np.save(f'{save_folder}/reward_list.npy', model.reward_list)
np.save(f'{save_folder}/delivered_list.npy', model.order_delivered_list)
np.save(f'{save_folder}/time_list.npy', model.order_time_list)
np.save(f'{save_folder}/dist_list.npy', model.order_dist_list)
np.save(f'{save_folder}/total_dist.npy', model.total_avg_dist)
np.save(f'{save_folder}/total_avg_time.npy', model.total_avg_time)
np.save(f'{save_folder}/q_table.npy', model.Q)
np.savetxt(f'{save_folder}/q_table_tracker.csv', model.Q_tracker, delimiter=',')
np.savetxt(f'{save_folder}/q_table.csv', model.Q, delimiter=',')


