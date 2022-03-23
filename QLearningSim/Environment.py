
import os, sys
sys.path.append(os.getcwd())
from collections import deque
import random
import numpy as np
from QLearningSim.Car import Car
import pandas as pd

class SimulationEnvironment:
    MAX_QUEUE_LENGTH = 3
    REWARD = 10
    def __init__(self, grid_length, restaurants, couriers, num_lt_bins, num_ot_bins, bin_size=None):
        self.starting_couriers = couriers
        self.grid_length = grid_length
        self.load_restaurants(restaurants)
        self.state = self.reset()
        self.total_order_count = 0
        self.total_order_time = 0
        self.total_order_distance = 0
        self.bin_size = bin_size
        self.num_lt_bins = num_lt_bins
        self.num_ot_bins = num_ot_bins
        self.lt_decision_lvls, self.ot_decision_lvls = self.get_decision_lvls()

    def reset(self):
        ## Reset entire environment
        self.load_couriers(self.starting_couriers)
        self.order_delivered = 0
        self.order_time = 0
        self.order_distance = 0
        self.C = 0
        self.timestep = 0
        self.order_queue = deque([])

        return [0, 0, 0, 0, 0]
    
    def l_t_bin(self, l_t):
        for i, lvl in enumerate(self.lt_decision_lvls):
            if l_t<=lvl:
                return i
        return self.num_lt_bins-1
    
    def o_t_bin(self, o_t):
        for i, lvl in enumerate(self.ot_decision_lvls):
            if o_t<=lvl:
                return i
        return self.num_ot_bins-1

    def get_state(self):
        '''
        Return current state and reward
        '''
        l_t_bins = []
        if self.total_order_count > 0:
            for courier in self.couriers.values():
                l_t_bins.append(self.l_t_bin(courier.queue_distance))
        else:
            for courier in self.couriers.values():
                l_t_bins.append(0)

        o_t_bins = []
        if len(self.order_queue) > 0 and self.total_order_count > 0:
            first_order = self.order_queue[0]
            for courier in self.couriers.values():
                o_t_bins.append(self.o_t_bin(courier.order_dist_from_last_queue(first_order[0], first_order[1])))
        else:
            for courier in self.couriers.values():
                o_t_bins.append(0)

        return [*l_t_bins, *o_t_bins, self.C]

    def step(self, action):
        '''
        Perform action that gets passed in
        action = 0 or 1
        '''
        ##prob some syntax errors, just a heads up
        ## Assign order to courier #action

        ## Update L_t, O_t, C
        courier = self.couriers[action]
        order = self.order_queue.popleft()
        
        ##get the restaurant and the house
        r = order[0]
        h = order[1]
        courier.add_order(r,h)
        
        l_t = courier.queue_distance
        ##calculate total order time
        T = l_t / courier.speed
        self.order_time += T
        self.total_order_time += T
        self.C = 1
        if self.total_order_count > 0:
            ## Dynamic reward function
            if T > 40:
                self.C = 0
        
        return [self.get_state(), SimulationEnvironment.REWARD if self.C == 1 else -2*SimulationEnvironment.REWARD]
    
    def get_actions(self):
        action_list = []
        for i, courier in self.couriers.items():
            if courier.get_queue_length() < 3:
                action_list.append(i)     
        return action_list

    def load_restaurants(self, restaurants: list):
        self.restaurants = {}
        for i in range(len(restaurants)):
            self.restaurants[i] = restaurants[i]
        
        all_locations = []
        for i in range(self.grid_length):
            for j in range(self.grid_length):
                all_locations.append((i, j))

        self.houses = {}
        count = 0
        for loc in all_locations:
            if loc in self.restaurants.values():
                continue      
            self.houses[count] = loc
            count += 1
    
    def load_couriers(self, couriers: list):
        self.couriers = {}
        for i in range(len(couriers)):
            self.couriers[i] = Car(self,couriers[i][1])
            self.couriers[i].update_location(couriers[i][0])

    def timestep_orders(self, order_rate):
        '''
        This will be called by an external "driver"
        '''

        orders = self.generate_orders_for_timestep(order_rate)
        for order in orders:
            self.order_queue.append(order)
    
    def timestep_deliveries(self):
        ## Update courier positions/deliveries
        for courier in self.couriers.values():
            courier.perform_deliveries()


    
    def generate_orders_for_timestep(self, order_rate):
        '''
        Generates random orders using possion clock for each house. 
        Randomly chooses restaurant to order from
        '''
        ## If queue length > 3 deny orders
        orders = []
        num_restaurant = len(self.restaurants)
        for house in self.houses:
            num_house_orders = np.random.poisson(order_rate / (self.grid_length ** 2 - num_restaurant))
            for i in range(num_house_orders):
                orders.append((self.restaurants[random.randint(0, num_restaurant - 1)], self.houses[house]))

        return orders
    
    def get_decision_lvls(self):
        lt_vals = pd.read_csv("QLearningSim/l_t.csv",index_col=0).values
        ot_vals = pd.read_csv("QLearningSim/o_t.csv",index_col=0).values
        return np.quantile(lt_vals,[1/(self.num_lt_bins)*i for i in range(1,self.num_lt_bins)]), np.quantile(ot_vals,[1/(self.num_ot_bins)*i for i in range(1,self.num_ot_bins)])


### Track number of visits to states
### Fix binning on o_t
### L_t bin with number of orders to

