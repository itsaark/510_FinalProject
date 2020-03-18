from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sns
from SSC import SSC

#Generating synthetic data
D = 3 #Ambient space dimention
d1 = 1
d2 = 1
N1 = 20
N2 = 20

X1 = np.random.randn(D,d1) @ np.random.randn(d1,N1)
X2 = np.random.randn(D,d2) @ np.random.randn(d2,N2)

d3 = 2
N3 = 50
X3 = np.random.randn(D,d3) @ np.random.randn(d3,N3)

X = np.vstack([X1.T,X2.T,X3.T])
C = SSC(X.T,800,800,800,200)
error = np.linalg.norm(X - (np.matmul(X.T,C)).T)
print("error is, ", error)



#plotting


# fig = plt.figure()
# ax = Axes3D(fig)
#
# ax.scatter(X1[0,:],X1[1,:],X1[2,:],'r')
# ax.scatter(X2[0,:],X2[1,:],X2[2,:],'g')
# ax.scatter(X3[0,:],X3[1,:],X3[2,:],'b')
# plt.show()
