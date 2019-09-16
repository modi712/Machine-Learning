import pandas as pd
import numpy as np
url='/home/nihar/Desktop/proj/haberman.csv'
df=pd.read_csv(url)
data=df.values
y = (data[:,-1]) - 1
x = (data[:,0:-1])/100.0
x[:,2] = x[:,2]*10.0/3.0
a = len(x[0])
print(a)
theta = 2*(np.random.rand(a))-1
print(x.shape)
print(theta.shape)
htheta = 1/(1+np.exp(-(np.matmul(x, theta))))
print(x.shape)
print(htheta.shape)
print(y.shape)
#print(htheta)

for i in range(100000):
	theta = theta + 0.01*np.transpose((np.matmul((np.transpose(y - htheta)),x)))
	htheta = 1/(1+np.exp(-(np.matmul(x, theta))))

print(htheta-y)
print(np.matmul(x, theta))
