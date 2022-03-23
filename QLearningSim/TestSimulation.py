from Environment import SimulationEnvironment
from ReadQtable import readQLearningModel
import numpy as np
from statistics import mean
import json

FOLDER = 'test'
NUM_EPISODES_TEST = 1000
TEST_TYPE = 0
save = False

with open(f'{FOLDER}/config.json') as f:
    config = json.load(f)

avg_order_dist = np.load(f'{FOLDER}/total_dist.npy')[-1]

sim = SimulationEnvironment(config['GRID_SIZE'], config['RESTAURANTS'], config['COURIERS'], config['NUM_LT_BINS'], config['NUM_OT_BINS'], bin_size=avg_order_dist)
model = readQLearningModel(NUM_EPISODES_TEST, config['NUM_TIMESTEPS_PER_DAY'], sim, config['ORDER_RATE'], TEST_TYPE,  visualize=False, folder=FOLDER)
model.run_sim()



print("The average reward is: " + str(mean(model.reward_list))) 
print("The average daily orders delivered is: " + str(mean(model.order_delivered_list)))
print(f'The average order time was: {model.total_avg_time[-1]}')
print(f'The average orders declined were: {mean(model.orders_declined_list)}')
print(f'The average orders acceptted were: {mean(model.orders_accepted_list)}')
print(f'The average orders were: {mean(model.total_orders_list)}')
if save == True:
    np.save(f'Testing_{TEST_TYPE}/reward_list.npy', model.reward_list)
    np.save(f'Testing_{TEST_TYPE}/delivered_list.npy', model.order_delivered_list)
    np.save(f'Testing_{TEST_TYPE}/time_list.npy', model.order_time_list)
    np.save(f'Testing_{TEST_TYPE}/dist_list.npy', model.order_dist_list)
    np.save(f'Testing_{TEST_TYPE}/total_dist.npy', model.total_avg_dist)
    np.save(f'Testing_{TEST_TYPE}/total_avg_time.npy', model.total_avg_time)