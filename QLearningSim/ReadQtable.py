import numpy as np
import pandas as pd

class readQLearningModel:
    def __init__(self, learning_rate, discount_rate, num_episodes, timesteps_per_day, environment, order_rate, sim_version):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_episodes = num_episodes
        self.reward_list = []
        self.timesteps_per_day = timesteps_per_day
        self.Q = pd.read_csv('q_table1.csv').to_numpy()
        self.environment = environment
        self.episode = 0
        self.order_rate = order_rate
        self.create_mapping()
        self.order_delivered_list = []
        self.sim_version = sim_version
        self.order_distance_list = []
        self.distance_from_last_order_list = []

    def run_sim(self):
        for day in range(self.num_episodes):
            print(f'day: {day}', end='\r')
            total_rewards = 0
            state = self.environment.reset()
            for j in range(self.timesteps_per_day):
                self.environment.timestep_orders(self.order_rate)
                num_orders = len(self.environment.order_queue)

                state = self.environment.get_state()
                for i in range(num_orders):
                    ## Check possible actions and limit action selection to those
                    possible_action = self.environment.get_actions()
                    if len(possible_action) > 0:
                        ## Indexing here needs to be fixed... QTable is 162 elements long but state space 
                        ## is only 5 elements long so we need some function to convert the indexes to work
                        ## properly. f(state) -> index of state
                        ## [1, 1, 0, 0, 1]
                        state_index = self.state_map[tuple(state)]
                        #prself.randomlf.Q[state_index,possible_action])
                        if self.sim_version==0:
                            action = np.argmax(self.Q[state_index,possible_action] + np.random.randn(1,2)*(1/(day+1))) if len(possible_action) > 1 else possible_action[0]
                        elif self.sim_version == 1:
                            action = min(self.couriers.items(), key=lambda x:x[1].queue_distance)[0] if len(possible_action) > 1 else possible_action[0]
                        elif self.sim_version == 2:
                            first_order = self.environment.order_queue[0]
                            action = min(self.couriers.items(), key=lambda x:x[1].order_dist_from_last_queue(first_order[0], first_order[0]))[0]
                        else:
                            action = np.random.randint(0,2) if len(possible_action) > 1 else possible_action[0]
                        next_state, reward = self.environment.step(action)
                        next_state_index = self.state_map[tuple(next_state)]
                        total_rewards += reward
                        state = next_state
                        self.order_distance_list.append(self.environment.couriers[0].queue_distance)                                                                      
                        self.order_distance_list.append(self.environment.couriers[1].queue_distance)
                        if len(self.environment.order_queue) > 0:
                            first_order = self.environment.order_queue[0]
                            self.distance_from_last_order_list.append(self.environment.couriers[0].order_dist_from_last_queue(first_order[0], first_order[1]))
                            self.distance_from_last_order_list.append(self.environment.couriers[1].order_dist_from_last_queue(first_order[0], first_order[1]))
                        #else:
                        #   self.distance_from_last_order_list.append(0)

                self.environment.timestep_deliveries()
            
            self.reward_list.append(total_rewards)
            self.order_delivered_list.append(self.environment.order_delivered)
    
    def create_mapping(self):
        self.state_map = {}
        count = 0
        for l1 in range(3):
            for l2 in range(3):
                for o1 in range(3):
                    for o2 in range(3):
                        for c in range(2):
                            state = [l1, l2, o1, o2, c]
                            self.state_map[tuple(state)] = count
                            count += 1
                
