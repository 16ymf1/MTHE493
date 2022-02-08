#Initialize table with all zeros
import numpy as np

#get environment from the other team
env = gym.make('INSERT NAME HERE')

#Q size should be 162 x 2
Q = np.zeros([env._observation_spec.n,env._action_spec.n])  #initialise Q table [env x actionspace]
# Set learning parameters
learning_rate = .8 #look at yu
discount_rate = .95 #discount rate
num_episodes = 2000
orders_in_day= 99
#create lists to contain total rewards and steps per episode
#jList = []
reward_list = []
for day in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    total_rewards = 0
    terminated = False
    order = 0
    #The Q-Table learning algorithm
    while order < orders_in_day:
        order+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env._action_spec.n)*(1./(day+1)))  #as more days pass we explore less
        #Get new state and reward from environment
        next_state, reward, terminated, info = env._step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + learning_rate*(reward + discount_rate*np.max(Q[next_state,:]) - Q[s,a])
        total_rewards += reward
        s = next_state
        if terminated == True:
            break
    #jList.append(j)
    reward_list.append(total_rewards)
print(reward_list)
