from Environment import SimulationEnvironment

NUM_TIMESTEPS = 10000
ORDER_RATE = 0.1

restuarants = [(0,0),(9,9),(4,3),(7,2),(3,8)]
couriers = [[(0,0), 1], [(2,3), 1]]
sim = SimulationEnvironment(10, restuarants, couriers, QLearningModel=None)
for i in range(NUM_TIMESTEPS):
    sim.timestep(ORDER_RATE)
