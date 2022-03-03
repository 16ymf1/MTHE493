import matplotlib.pyplot as plt
import numpy as np

values = np.load('reward_list.npy')
delivered = np.load('delivered_list.npy')
avgQueueOrder = np.load('average_order_list.npy')
avgOrder = np.load('server_order_distance_list.npy')


plt.plot(values)
plt.xlabel('Days Simulated')
#plt.ylabel('Reward')
plt.show()

plt.plot(delivered)
plt.xlabel('Days Simulated')
#plt.ylabel('Orders Delivered')
plt.show()



arr_avgOrderQueue = np.array(avgQueueOrder)
print('Min order queue distances is: ' + str(arr_avgOrderQueue.min()))
print('33 percentile of order queue distances is: ' + str(np.quantile(arr_avgOrderQueue,0.33)))
print('67 percentile of order queue distances is: ' + str(np.quantile(arr_avgOrderQueue,0.67)))
print('Max order queue distances is: ' + str(arr_avgOrderQueue.max()))
print('Average order queue distances is: ' + str(arr_avgOrderQueue.mean()))

arr_avgNewOrder = np.array(avgOrder)
print('Min order distances is: ' + str(arr_avgNewOrder.min()))
print('33 percentile of order distances is: ' + str(np.quantile(arr_avgNewOrder,0.33)))
print('67 percentile of order distances is: ' + str(np.quantile(arr_avgNewOrder,0.67)))
print('Max order distances is: ' + str(arr_avgNewOrder.max()))
print('Average order distances is: ' + str(arr_avgNewOrder.mean()))