from Order_Simulator import Order_Simulator

sim = Order_Simulator(4, 2, 2, 10)
for i in range(10):
    print(f'Timestep: {i}')
    sim.track_variables()