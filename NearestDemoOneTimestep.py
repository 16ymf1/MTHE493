from OrderSimulator import Order_Simulator

sim = Order_Simulator(4, 2, 2, 5)
print('----------------------------------------------------------------')
sim.visualize_layout()
for i in range(2):
    print(f'Timestep: {i}')
    sim.simple_simulation(visualize=True, timestep=i)
    print('----------------------------------------------------------------')