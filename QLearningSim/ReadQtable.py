import numpy as np
import pandas as pd
import itertools

class readQLearningModel:
    def __init__(self, num_episodes, timesteps_per_day, environment, order_rate, sim_version, visualize=False, folder=None):
        self.num_episodes = num_episodes
        self.reward_list = []
        self.timesteps_per_day = timesteps_per_day

        self.Q = np.load(f'{folder}/q_table.npy')
        self.environment = environment
        self.episode = 0
        self.order_rate = order_rate
        self.num_lt_bins = environment.num_lt_bins
        self.num_ot_bins = environment.num_ot_bins
        self.create_mapping()
        self.order_delivered_list = []
        self.sim_version = sim_version
        self.order_time_list = []
        self.order_dist_list = []
        self.total_avg_dist = []
        self.total_avg_time = []
        self.orders_declined_list = []
        self.orders_accepted_list = []
        self.total_orders_list = []
        self.visualize = visualize

    def run_sim(self):
        for day in range(self.num_episodes):
            print(f'day: {day}', end='\r')
            total_rewards = 0
            orders_declined = 0
            orders_accepted = 0
            total_orders = 0
            state = self.environment.reset()
            for j in range(self.timesteps_per_day):
                self.environment.timestep_orders(self.order_rate)
                num_orders = len(self.environment.order_queue)
                total_orders += num_orders

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
                            action = np.argmax(self.Q[state_index,possible_action]) if len(possible_action) > 1 else possible_action[0]
                        elif self.sim_version == 1:
                            action = min(self.environment.couriers.items(), key=lambda x:x[1].queue_distance)[0] if len(possible_action) > 1 else possible_action[0]
                        elif self.sim_version == 2:
                            first_order = self.environment.order_queue[0]
                            action = min(self.environment.couriers.items(), key=lambda x:x[1].order_dist_from_last_queue(first_order[0], first_order[0]))[0]
                        else:
                            action = np.random.randint(0,2) if len(possible_action) > 1 else possible_action[0]
                        next_state, reward = self.environment.step(action)
                        if self.visualize:
                            print(f'Timestep: {j}')
                            print(f'Courier 1: {self.environment.couriers[0].get_queue_length()}')
                            print(f'Courier 2: {self.environment.couriers[1].get_queue_length()}')
                            print(f'Action: {action}, Reward: {reward}')
                        total_rewards += reward
                        state = next_state
                        orders_accepted += 1
                    else:
                        total_rewards -= 20
                        orders_declined += 1
                self.environment.timestep_deliveries()
            
            self.reward_list.append(total_rewards)
            self.orders_declined_list.append(orders_declined)
            self.total_orders_list.append(total_orders)
            self.order_delivered_list.append(self.environment.order_delivered)
            self.orders_accepted_list.append(orders_accepted)
            self.order_time_list.append(self.environment.order_time / self.environment.order_delivered)
            self.order_dist_list.append(self.environment.order_distance / self.environment.order_delivered)
            self.total_avg_dist.append(self.environment.total_order_distance / self.environment.total_order_count)
            self.total_avg_time.append(self.environment.total_order_time / self.environment.total_order_count)

    
    def create_mapping(self):
        self.state_map = {}

        num_couriers = len(self.environment.couriers)

        l_t_states = [[*range(self.num_lt_bins)]] * num_couriers
        o_t_states = [[*range(self.num_ot_bins)]] * num_couriers
        c_states = [0,1]

        all_states = itertools.product(*l_t_states, *o_t_states, c_states)
        for i, state in enumerate(all_states):
            self.state_map[state] = i
                
