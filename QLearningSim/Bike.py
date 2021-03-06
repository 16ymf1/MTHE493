from Courier import Courier
import numpy as np

class Bike(Courier):
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
        order_distance = np.linalg.norm([curr_x - r_x, curr_y - r_y]) + np.linalg.norm([r_x - h_x, r_y - h_y])
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
        order_distance = np.linalg.norm([curr_x - r_x, curr_y - r_y]) + np.linalg.norm([r_x - h_x, r_y - h_y])
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
            if visualize:
                print('No orders in queue')
            return
        
        if visualize:
            print('Start:')
        add = 1 if self.curr_order else 0
        t = timestep if timestep is not None else 't'
        if visualize is not None:
            print(f'L_{t} = {self.queue_distance}, A_{t} = {self.new_distance}, S_{t} = {self.speed}, Queue Length: {len(self.order_queue) + add}')

        
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
                if visualize:
                    print(f'Completed order - Started: {starting_pos} to Restaurant: {self.curr_order[0]} to House: {self.curr_order[1]}')
                    print(f'Distance covered: {dist_left_in_order}')
                    self.simulation_instance.visualize_layout()
                    print('---------------------------------------------------------------------')
                    
                self.curr_order = None
                if len(self.order_queue) > 0:
                    self.curr_order = self.order_queue.popleft()
            else:
                dist_to_restaurant = self.order_dist_from_current_loc(
                    self.curr_order[0], self.curr_order[0])
                dist_covered = timestep_dist
                ## If distance to restaurant greater than timestep distance
                ## move as close as possible to restaurant
                if dist_to_restaurant > timestep_dist:
                    self.partial_move(timestep_dist, 0)
                ## Else, move to restaurant, then as close as possible to delivery house
                else:
                    timestep_dist -= dist_to_restaurant
                    self.location = self.curr_order[0]
                    old_curr = self.curr_order
                    self.curr_order = (self.curr_order[1], self.curr_order[1])
                    self.partial_move(timestep_dist, 1)
                
                if visualize:
                    print(
                        f'Partially completed order - Starting: {starting_pos} to Restaurant: {old_curr[0]} to House: {old_curr[1]}')
                    print(f'Made it to: {self.location}, distance covered: {dist_covered}')
                    self.simulation_instance.visualize_layout()
                    print('-------------------------------------------------------------')
                
                timestep_dist = 0
        
        add = 1 if self.curr_order else 0
        
        if visualize:
            print('End:')
            print(f'L_{t} = {self.queue_distance}, A_{t} = {self.new_distance}, S_{t} = {self.speed}, Queue Length: {len(self.order_queue) + add}')
    
    def partial_move(self, timestep_dist, i):
        dst_x, dst_y = self.curr_order[i]
        curr_x, curr_y = self.location
        x_dir, y_dir = dst_x - curr_x, dst_y - curr_y
        time = (self.speed / timestep_dist)
        self.location = (curr_x + x_dir * (self.speed / timestep_dist), curr_y + y_dir * (self.speed / timestep_dist))



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

        return timestep_dist

if __name__ == '__main__':
    c = Bike(None, 10, (0,0))
    print(c.order_dist_from_current_loc((1,1),(2,2)))