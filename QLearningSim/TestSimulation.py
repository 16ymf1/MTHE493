from Environment import SimulationEnvironment
from ReadQtable import readQLearningModel
import numpy as np
from statistics import mean
  

ORDER_RATE = 0.08
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.95
NUM_EPISODES = 1000
NUM_TIMESTEPS = 288
RANDOM=3
save = False

restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]

avg_order_dist = np.load('twoThirdResults/total_dist.npy')[-1]

sim = SimulationEnvironment(10, restuarants, couriers, bin_size=avg_order_dist)
model = readQLearningModel(LEARNING_RATE, DISCOUNT_RATE, NUM_EPISODES, NUM_TIMESTEPS, sim, ORDER_RATE, RANDOM, visualize=False)
model.run_sim()



print("The average reward is: " + str(mean(model.reward_list))) 
print("The average daily orders delivered is: " + str(mean(model.order_delivered_list)))
print(f'The average order time was: {model.total_avg_time[-1]}')
print(f'The average orders declined were: {mean(model.orders_declined_list)}')
print(f'The average orders acceptted were: {mean(model.orders_accepted_list)}')
print(f'The average orders were: {mean(model.total_orders_list)}')
if save == True:
    np.save(f'Testing_{RANDOM}/reward_list.npy', model.reward_list)
    np.save(f'Testing_{RANDOM}/delivered_list.npy', model.order_delivered_list)
    np.save(f'Testing_{RANDOM}/time_list.npy', model.order_time_list)
    np.save(f'Testing_{RANDOM}/dist_list.npy', model.order_dist_list)
    np.save(f'Testing_{RANDOM}/total_dist.npy', model.total_avg_dist)
    np.save(f'Testing_{RANDOM}/total_avg_time.npy', model.total_avg_time)