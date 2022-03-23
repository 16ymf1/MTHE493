import numpy as np
import random
import itertools

class QLearningModel:
    def __init__(self, learning_rate, discount_rate, num_episodes, timesteps_per_day, environment, order_rate, num_lt_bins=3, num_ot_bins=3,track_cells=False, q_table=None):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = 0.9
        self.num_episodes = num_episodes
        self.reward_list = []
        self.timesteps_per_day = timesteps_per_day
        self.environment = environment
        num_couriers = len(environment.couriers)
        state_count = (num_lt_bins ** num_couriers) * (num_ot_bins ** num_couriers) * 2
        self.num_lt_bins = num_lt_bins
        self.num_ot_bins = num_ot_bins
        if q_table:
            self.Q = np.load(q_table)
        else:  
            self.Q = np.zeros((state_count, num_couriers))
        self.Q_tracker = np.zeros((state_count,num_couriers))
        self.episode = 0
        self.order_rate = order_rate
        self.create_mapping()
        self.order_delivered_list = []
        self.order_time_list = []
        self.order_dist_list = []
        self.total_avg_dist = []
        self.total_avg_time = []
        self.cell_tracker = [[[] for i in range(num_couriers)] for i in range(state_count)]
        self.lt_vals = []
        self.ot_vals = []
        self.track_cells = track_cells

    def run_sim(self):
        for day in range(self.num_episodes):
            print(f'day: {day}', end='\r')
            total_rewards = 0
            state = self.environment.reset()
            for j in range(self.timesteps_per_day):
                self.environment.timestep_orders(self.order_rate)
                num_orders = len(self.environment.order_queue)
                orders = self.environment.order_queue.copy()

                state = self.environment.get_state()
                for i in range(num_orders):
                    ## Check possible actions and limit action selection to those
                    possible_action = self.environment.get_actions()
                    if len(possible_action) > 0:
                        state_index = self.state_map[tuple(state)]
                        if random.random() < self.epsilon:
                            action = np.argmax(self.Q[state_index,possible_action]) if len(possible_action) > 1 else possible_action[0]
                        else:
                            action = random.choice(possible_action)

                        next_state, reward = self.environment.step(action)
                        next_state_index = self.state_map[tuple(next_state)]
                        state_count = self.Q_tracker[state_index, action]
                        self.Q[state_index,action] = self.Q[state_index,action] + (1/(1 + (1/10)*state_count))*(reward + self.discount_rate*np.max(self.Q[next_state_index,:]) - self.Q[state_index,action])
                        self.Q_tracker[state_index, action] = self.Q_tracker[state_index, action] + 1
                        if self.track_cells:
                            for i in self.environment.couriers:
                                self.cell_tracker[state_index][i].append(self.Q[state_index, i])
                        total_rewards += reward
                        state = next_state
                if num_orders>0:
                    for i in range(len(self.environment.couriers)):
                        self.lt_vals.append(self.environment.couriers[i].queue_distance)
                        self.ot_vals.append(self.environment.couriers[i].order_dist_from_last_queue(orders[-1][0],orders[-1][1]))
                self.environment.timestep_deliveries()
            
            self.reward_list.append(total_rewards)
            self.order_delivered_list.append(self.environment.order_delivered)
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
        

                

