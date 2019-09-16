import pandas as pd
import numpy as np
url='/home/nihar/Desktop/proj/haberman.csv'
df=pd.read_csv(url)
data=df.values
y = (data[0:303,-1]) 
x = (data[0:303,0:-1])

a = x.shape
print(a)

x1 = x[0:101,:]
x2 = x[101:202,:]
x3 = x[202:303,:]

m1 = np.array((30, 40, 10))
m2 = np.array((31, 42, 3))


print(m1)


print(m1.shape)
#print(x11.shape)


for j in range(50):
	x11 = np.zeros((1,3))
	x12 = np.zeros((1,3))
	

	d = np.zeros((1,2))

	for i in range(len(x)):
		d[0,0] = np.linalg.norm(m1 - x[i])
		d[0,1] = np.linalg.norm(m2 - x[i])
			 
		if d[0,0] < d[0,1]: 
			x11 = np.vstack((x11,x[i]))
		if d[0,1] < d[0,0]: 
			x12 = np.vstack((x12,x[i]))
		
	m1 = x11.mean(axis=0)
	m2 = x12.mean(axis=0)
	
	print(m1)
	print(m2)

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

x = x11[:,0]
y = x11[:,1]
z = x11[:,2]
x, y, z = np.broadcast_arrays(x, y, z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z)


x = x12[:,0]
y = x12[:,1]
z = x12[:,2]
x, y, z = np.broadcast_arrays(x, y, z)

fig = plt.figure()
ay = fig.add_subplot(111, projection='3d')
ay.scatter(x,y,z)
plt.show()

	






