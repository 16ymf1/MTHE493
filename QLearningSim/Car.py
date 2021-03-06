from Courier import Courier
import numpy as np

class Car(Courier):
    def __init__(self, simulation_instance, courier_speed, initial_pos=None):
        super().__init__(simulation_instance, courier_speed, initial_pos)

    def order_dist_from_current_loc(self, restaurant, house):
        '''
        Computes distance from current location to the end of the order
        from the current courier location
        '''
        curr_x, curr_y = self.location
        r_x, r_y = restaurant
        h_x, h_y = house
        order_distance = abs(curr_x - r_x) + abs(curr_y - r_y) + abs(r_x - h_x) + abs(r_y - h_y)
        return order_distance
    
    def order_dist_from_last_queue(self, restaurant, house):
        '''
        Computes distance from current location to the end of the order
        from the last order in the queue
        '''
        if len(self.order_queue) > 0:
            curr_x, curr_y = self.order_queue[-1][-1]
        elif self.curr_order:
            curr_x, curr_y = self.curr_order[-1]
        else:
            curr_x, curr_y = self.location
        r_x, r_y = restaurant
        h_x, h_y = house
        order_distance = abs(curr_x - r_x) + abs(curr_y - r_y) + abs(r_x - h_x) + abs(r_y - h_y)
        return order_distance

    def perform_deliveries(self, visualize=False, timestep=None):
        '''
        Performs the deliveries for a timestep. Will partially
        complete orders that do not have enough time to be fully
        completed
        '''

        ## Distance covered by courier in one timestep
        timestep_dist = self.speed * 1

        ## If there is no current order, nothing to do
        if not self.curr_order:
            return
   
        ## Updating distance tracking variables
        if self.queue_distance > timestep_dist:
            self.queue_distance -= timestep_dist
        else:
            self.queue_distance = 0
        self.new_distance = 0

        ## Deliver orders while there is still distance left in the timestep, and
        ## a currently active order
        while timestep_dist > 0 and self.curr_order:
            dist_left_in_order = self.order_dist_from_current_loc(*self.curr_order)
            ## If order fully completable then update locations and pop next order
            ## from queue if it exists
            starting_pos = self.location
            if dist_left_in_order <= timestep_dist:
                self.location = self.curr_order[1]
                timestep_dist -= dist_left_in_order
                    
                self.curr_order = None
                if len(self.order_queue) > 0:
                    self.curr_order = self.order_queue.popleft()
                    self.simulation_instance.order_delivered += 1
            else:
                dist_to_restaurant = self.order_dist_from_current_loc(self.curr_order[0], self.curr_order[0])
                ## If distance to restaurant greater than timestep distance
                ## move as close as possible to restaurant
                if dist_to_restaurant > timestep_dist:
                    self.partial_move(timestep_dist, 0)
                ## Else, move to restaurant, then as close as possible to delivery house
                else:
                    timestep_dist -= dist_to_restaurant
                    self.location = self.curr_order[0]
                    self.curr_order = (self.curr_order[1], self.curr_order[1])
                    self.partial_move(timestep_dist, 1)
                
                timestep_dist = 0
    
    def partial_move(self, timestep_dist, i):
        dst_x, dst_y = self.curr_order[i]
        curr_x, curr_y = self.location
        x_dist = dst_x - curr_x
        ## First move along x-axis
        if abs(x_dist) > timestep_dist:
            self.location = (
                curr_x + timestep_dist * np.sign(x_dist), curr_y)
            timestep_dist = 0
        else:
            timestep_dist -= abs(x_dist)
            self.location = (curr_x + x_dist, curr_y)
        ## Second move along y-axis
        curr_x, curr_y = self.location
        y_dist = dst_y - curr_y
        self.location = (curr_x, curr_y + timestep_dist * np.sign(y_dist))