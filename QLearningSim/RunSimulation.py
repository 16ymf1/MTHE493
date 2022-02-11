from Environment import SimulationEnvironment
from QLearningModel import QLearningModel
import numpy as np

ORDER_RATE = 0.15
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.95
NUM_EPISODES = 100000
NUM_TIMESTEPS = 288


restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]
sim = SimulationEnvironment(10, restuarants, couriers)
model = QLearningModel(LEARNING_RATE, DISCOUNT_RATE, NUM_EPISODES, NUM_TIMESTEPS, sim, ORDER_RATE)
model.run_sim()

np.save('reward_list.npy', model.reward_list)
np.save('delivered_list.npy', model.order_delivered_list)
np.save('q_table.npy', model.Q)
np.savetxt('q_table.csv', model.Q, delimiter=',')


