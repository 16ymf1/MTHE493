
import os, sys
from re import A
sys.path.append(os.getcwd())
from collections import deque
import random
import numpy as np
#from QLearningSim.Car import Car
from Car import Car
import math

class SimulationEnvironment:
    MAX_QUEUE_LENGTH = 3
    REWARD = 10
    def __init__(self, grid_length, restaurants, couriers):
        self.starting_couriers = couriers
        self.grid_length = grid_length
        self.load_restaurants(restaurants)
        self.state = self.reset()

    def reset(self):
        ## Reset entire environment
        self.load_couriers(self.starting_couriers)
        self.order_delivered = 0
        self.C = 0
        self.timestep = 0
        self.order_queue = deque([])

        return [0, 0, 0, 0, 0]

    def get_state(self):
        '''
        Return current state and reward
        '''
        l_t1 = math.floor(self.couriers[0].queue_distance / (self.grid_length * 4))
        l_t2 = math.floor(self.couriers[1].queue_distance / (self.grid_length * 4))

        if len(self.order_queue) > 0:
            first_order = self.order_queue[0]
            o_t1 = math.floor(self.couriers[0].order_dist_from_last_queue(first_order[0], first_order[1]) / ((self.grid_length * 4) / 3))
            o_t2 = math.floor(self.couriers[1].order_dist_from_last_queue(first_order[0], first_order[1]) / ((self.grid_length * 4) / 3))
        else:
            o_t1 = 0
            o_t2 = 0

        return [l_t1, l_t2, o_t1, o_t2, self.C]

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
        T= l_t / courier.speed
        self.C = 0
        if T <= 45:
            self.C = 1
        
        return [self.get_state(), SimulationEnvironment.REWARD if self.C == 1 else -SimulationEnvironment.REWARD]
    
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

    