import random
import numpy as np

class Order_Simulator():

    def __init__(self, grid_length, num_restaurants, num_drivers):
        self.grind_length = grid_length
        self.num_restaurants = num_restaurants
        self.num_drivers = num_drivers
        self.generate_restaurants_and_houses()
        self.last_step =  24   # to simulate one day, with 5 min intervals
        self.rate = 245.0 / 24

    def generate_restaurants_and_houses(self): 
        self.restaurants = {} 
        for i in range(self.num_restaurants):
            restaurant_location = (random.randint(0, self.grind_length - 1), random.randint(0, self.grind_length - 1))
            while restaurant_location in self.restaurants.values(): 
                restaurant_location = (random.randint(0, self.grind_length - 1), random.randint(0, self.grind_length - 1)) 
            self.restaurants[i] = restaurant_location
        
        all_locations = []
        for i in range(self.grind_length):
            for j in range(self.grind_length):
                all_locations.append((i, j))
        

        self.houses = {}
        count = 0
        for loc in all_locations:
            if loc in self.restaurants.values():
                continue      
            self.houses[count] = loc
            count += 1
    
    def generate_couriers(self):
        self.couriers = {}
        

    def visualize_layout(self): 

        arr = [[None] * self.grind_length for i in range(self.grind_length)]

        for restaurant in self.restaurants.values():
            arr[restaurant[0]][restaurant[1]] = 'R'
        
        for house in self.houses.values():
            arr[house[0]][house[1]] = 'H'
        
        for row in arr:
            print(row)

    def generate_orders_for_timestep(self):

        orders = []
        number_of_new_orders = np.random.poisson(self.rate)
        for new_order_number in range(number_of_new_orders):
          orders.append([random.choice([*self.restaurants.values()]),random.choice([*self.houses.values()])])

        return orders
      
if __name__ == "__main__":
    sim = Order_Simulator(4, 2, 2)
    sim.visualize_layout()
    for i in range(10):
        print(len(sim.generate_orders_for_timestep()))
    