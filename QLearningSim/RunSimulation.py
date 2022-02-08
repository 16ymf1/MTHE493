from Environment import SimulationEnvironment
from QLearningModel import QLearningModel

ORDER_RATE = 0.15
LEARNING_RATE = 0.8
DISCOUNT_RATE = 0.95
NUM_EPISODES = 2000
NUM_TIMESTEPS = 288


restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]
sim = SimulationEnvironment(10, restuarants, couriers)
model = QLearningModel(LEARNING_RATE, DISCOUNT_RATE, NUM_EPISODES, NUM_TIMESTEPS, sim, ORDER_RATE)
model.run_sim()

print(model.reward_list)


