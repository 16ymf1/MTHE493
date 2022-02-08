import numpy as np

class QLearningModel:
    def __init__(self, learning_rate, discount_rate, num_episodes, timesteps_per_day, environment, order_rate):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.num_episodes = num_episodes
        self.reward_list = []
        self.timesteps_per_day = timesteps_per_day
        self.environment = environment
        self.Q = np.zeros((2, 162))
        self.episode = 0
        self.order_rate = order_rate

    def run_sim(self):
        for day in range(self.num_episodes):
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
                        action = np.argmax(self.Q[state,possible_action] + np.random.randn(1,2)*(1./(day+1)))
                        next_state, reward = self.environment.step(action)
                        self.Q[state,action] = self.Q[state,action] + self.learning_rate*(reward + self.discount_rate*np.max(self.Q[next_state,:]) - self.Q[state,action])
                        total_rewards += reward
                        state = next_state

                self.environment.timestep_deliveries()
            
            self.reward_list.append(total_rewards)
