from OrderSimulator import Order_Simulator

sim = Order_Simulator(4, 2, 2, 10)
sim.visualize_layout()
for i in range(1):
    print(f'Timestep: {i}')
    sim.simple_simulation()