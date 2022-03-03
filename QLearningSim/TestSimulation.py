from Environment import SimulationEnvironment
from ReadQtable import readQLearningModel
import numpy as np
from statistics import mean
  

ORDER_RATE = 0.15
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.95
NUM_EPISODES = 1000
NUM_TIMESTEPS = 288
RANDOM=0

restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]
sim = SimulationEnvironment(10, restuarants, couriers)
model = readQLearningModel(LEARNING_RATE, DISCOUNT_RATE, NUM_EPISODES, NUM_TIMESTEPS, sim, ORDER_RATE, RANDOM)
model.run_sim()


print("the average reward is: " + str(mean(model.reward_list))) 
np.save('reward_list.npy', model.reward_list)
print("the average daily orders delivered is: " + str(mean(model.order_delivered_list))) 
np.save('delivered_list.npy', model.order_delivered_list)
np.save('average_order_list.npy', model.order_distance_list)
np.save('q_table.npy', model.Q)
np.savetxt('q_table.csv', model.Q, delimiter=',')
np.save('server_order_distance_list.npy',model.distance_from_last_order_list)