import os, sys
sys.path.append(os.getcwd())
from Environment import SimulationEnvironment
from QLearningModel import QLearningModel
import numpy as np
import json

# If you have a config file use this variable otherwise leave it None
CONFIG_FILE = None
# Customize a config below
config = {
    'GRID_SIZE': 10,
    'ORDER_RATE': 0.08,
    'LEARNING_RATE': 0.6,
    'DISCOUNT_RATE': 0.8,
    'NUM_EPISODES_TRAIN': 10000,
    'NUM_TIMESTEPS_PER_DAY': 288,
    'NUM_LT_BINS': 4,
    'NUM_OT_BINS': 4,
    'SAVE_FOLDER': 'test',
    'RESTAURANTS': [(0,0),(9,9),(4,3),(7,2),(3,8)],
    'COURIERS':[[(0,0), 2], [(2,3), 1], [(5,5), 1]]
    }

if CONFIG_FILE is not None:
    with open(CONFIG_FILE) as f:
        config = json.load(f)

if not os.path.isdir(config['SAVE_FOLDER']):
    os.mkdir(config['SAVE_FOLDER'])

with open(f'{config["SAVE_FOLDER"]}/config.json', 'w') as f:
        json.dump(config, f)

sim = SimulationEnvironment(config['GRID_SIZE'], config['RESTAURANTS'], config['COURIERS'], config['NUM_LT_BINS'], config['NUM_OT_BINS'], bin_size=14.267)
model = QLearningModel(config['LEARNING_RATE'], config['DISCOUNT_RATE'], config['NUM_EPISODES_TRAIN'], config['NUM_TIMESTEPS_PER_DAY'], sim, config['ORDER_RATE'], config['NUM_LT_BINS'], config['NUM_OT_BINS'], track_cells=True)
model.run_sim()

np.save(f'{config["SAVE_FOLDER"]}/reward_list.npy', model.reward_list)
np.save(f'{config["SAVE_FOLDER"]}/delivered_list.npy', model.order_delivered_list)
np.save(f'{config["SAVE_FOLDER"]}/time_list.npy', model.order_time_list)
np.save(f'{config["SAVE_FOLDER"]}/dist_list.npy', model.order_dist_list)
np.save(f'{config["SAVE_FOLDER"]}/total_dist.npy', model.total_avg_dist)
np.save(f'{config["SAVE_FOLDER"]}/total_avg_time.npy', model.total_avg_time)
np.save(f'{config["SAVE_FOLDER"]}/q_table.npy', model.Q)
np.save(f'{config["SAVE_FOLDER"]}/cell_tracker.npy', np.array(model.cell_tracker))
np.savetxt(f'{config["SAVE_FOLDER"]}/q_table_tracker.csv', model.Q_tracker, delimiter=',')
np.savetxt(f'{config["SAVE_FOLDER"]}/q_table.csv', model.Q, delimiter=',')



