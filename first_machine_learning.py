import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3D import Axes3D

#random input data to train on

observations =1000
xs= np.random.uniform(low=-10,high=10,size=(observations,1))
zs=np.random.uniform(-10,10,(observations,1))
inputs=np.column_stack((xs,zs))
#print(inputs.shape)

#creating targets for us
noise = np.random.uniform(-1,1,(observations,1))
targets= 2*xs-5*zs+noise
#print(targets.shape)

#plotting the training data
targets= targets.reshape(observations,)
fig=plt.figure()
ax= fig.add_subplot(111,projection='3d')
ax.plot(xs,zs,targets)
plt.show()
targets=targets.reshape(observations,1)
#initializing variables
init_range=0.1
weights=np.random.uniform(-init_range,init_range,size=(2,1))
biases=np.random.uniform(-init_range,init_range,size=1)
print(weights)
print(biases)

#setting a learning rate
learning_rate = 0.2
#train the model
for i in range(100):
    outputs=np.dot(inputs,weights)+biases
	deltas= outputs=targets
	loss=np.sum(deltas**2)/2/observations
	print(loss)
	deltas_scaled=deltas/observations
	weights=weights-learning_rate*np.dot(inputs.T,deltas_scaled)
	biases=biases-learning_rate*np.sum(deltas_scaled)
	
print(weights,biases)
#again plotting without labelling
plt.plot(outputs,targets)
plt.show()

