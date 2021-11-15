from math import dist
import random

class Courier:
    def __init__(self, simulation_instance, courier_speed):
        self.simulation_instance = simulation_instance
        self.location = self.initialize_position()
        self.order_queue = []
        self.queue_distance = 0
        self.speed = courier_speed
    
    def initialize_position(self):
        '''
        Initializes courier postion randomly on the grid
        '''
        return (random.randint(0, self.simulation_instance.grid_length - 1), random.randint(0, self.simulation_instance.grid_length - 1))
    
    def compute_order_distance(self, restaurant, house):
        '''
        Computes distance from current location to the end of the order
        '''
        curr_x, curr_y = self.location
        r_x, r_y = restaurant
        h_x, h_y = house
        order_distance = abs(curr_x - r_x) + abs(curr_y - r_y) + abs(r_x - h_x) + abs(r_y - h_y)
        return order_distance
    
    def update_location(self, new_location):
        self.location = new_location
    
    def add_distance(self, distance):
        self.queue_distance += distance
    
    def perform_deliveries(self):
        if self.queue_distance > self.speed:
            self.queue_distance -= self.speed
        else:
            self.queue_distance = 0
