import random
import numpy as np
from Courier import Courier

class Order_Simulator():

    def __init__(self, grid_length, num_restaurants, num_couriers, order_rate):
        self.grid_length = grid_length
        self.num_restaurants = num_restaurants
        self.num_couriers = num_couriers
        self.order_rate = order_rate
        self.initialize_restaurants_and_houses()
        self.initialize_couriers()

    def initialize_restaurants_and_houses(self):
        '''
        Initializes positions of restaurants and houses within grid
        '''
        self.restaurants = {} 
        for i in range(self.num_restaurants):
            restaurant_location = (random.randint(0, self.grid_length - 1), random.randint(0, self.grid_length - 1))
            while restaurant_location in self.restaurants.values(): 
                restaurant_location = (random.randint(0, self.grid_length - 1), random.randint(0, self.grid_length - 1)) 
            self.restaurants[i] = restaurant_location
        
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
    
    def initialize_couriers(self):
        '''
        Initialize courier objects 
        '''
        self.couriers = {}
        for i in range(self.num_couriers):
            self.couriers[i] = Courier(self, 15)
            print(f'Courier {i} initial L_t = 0')

    def generate_orders_for_timestep(self):
        '''
        Generates random orders using possion clock and assigning each order to a house/restaurant pair
        '''
        orders = []
        number_of_new_orders = np.random.poisson(self.order_rate)
        for i in range(number_of_new_orders):
            orders.append((random.randint(0, self.num_restaurants - 1),random.randint(0, self.grid_length ** 2 - self.num_restaurants - 1)))
        return orders
    
    def visualize_layout(self): 
        '''
        Prints out layout of houses/restaurants for visualization purposes
        '''
        arr = [[None] * self.grid_length for i in range(self.grid_length)]

        for restaurant in self.restaurants.values(): arr[restaurant[1]][restaurant[0]] = 'R'
      
        for house in self.houses.values(): arr[house[1]][house[0]] = 'H'

        for courier_num, courier in self.couriers.items():
            x, y = courier.location
            arr[y][x] += 'C' + str(courier_num)
            
        for row in arr: print(row)
    
    def simple_simulation(self, visualize=False, timestep=None):
        '''
        Simulation where orders are assigned randomly to either courier
        '''
        orders = self.generate_orders_for_timestep()
        print(f'There were {len(orders)} orders placed this timestep:')
        for i, order in enumerate(orders):
            restaurant, house = self.restaurants[order[0]], self.houses[order[1]]
            courier_num = random.randint(0, self.num_couriers - 1)
            if visualize:
                print(f'Order {restaurant} -> {house} assigned to courier {courier_num}')
            courier = self.couriers[courier_num]
            courier.add_order(restaurant, house)
        
        for i, courier in self.couriers.items():
            print(f'Courier {i}:')
            courier.perform_deliveries(visualize=visualize, timestep=timestep)
    
    def nearest_simulation(self, visualize=False, timestep=None):
        '''
        Simulation where orders are assigned randomly to either courier
        '''
        orders = self.generate_orders_for_timestep()
        print(f'There were {len(orders)} orders placed this timestep:')
        for i, order in enumerate(orders):
            restaurant, house = self.restaurants[order[0]], self.houses[order[1]]
            courier_num = min(self.couriers.items(), key=lambda x:x[1].distance_from_last_queue(restaurant, restaurant))[0]
            if visualize:
                print(f'Order {restaurant} -> {house} assigned to courier {courier_num}')
            courier = self.couriers[courier_num]
            courier.add_order(restaurant, house)
        
        for i, courier in self.couriers.items():
            print(f'Courier {i}:')
            courier.perform_deliveries(visualize=visualize, timestep=timestep)


    
