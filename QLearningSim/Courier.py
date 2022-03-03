import random
from collections import deque

class Courier:
    def __init__(self, simulation_instance, courier_speed, initial_pos=None):
        self.simulation_instance = simulation_instance
        self.location = initial_pos if initial_pos else self.initialize_position()
        self.order_queue = deque([])
        self.curr_order = None
        self.queue_distance = 0     # L_t
        self.new_distance = 0       # A_t
        self.speed = courier_speed  # S_t
    
    def initialize_position(self):
        '''
        Initializes courier postion randomly on the grid
        '''
        return (random.randint(0, self.simulation_instance.grid_length - 1), random.randint(0, self.simulation_instance.grid_length - 1))
    
    def update_location(self, new_location):
        self.location = new_location
    
    def get_queue_length(self):
        add = 1 if self.curr_order else 0
        return len(self.order_queue) + add
    
    def add_order(self, restaurant, house):
        '''
        Adds order to couriers queue
        '''
        dist = self.order_dist_from_last_queue(restaurant, house)
        self.simulation_instance.order_distance += dist
        self.simulation_instance.total_order_distance += dist
        self.simulation_instance.total_order_count += 1
        self.order_queue.append((restaurant, house))
        self.queue_distance += dist
        self.new_distance += dist
        if not self.curr_order:
            self.curr_order = self.order_queue.popleft()