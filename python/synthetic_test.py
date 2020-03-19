from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import helperFunctions as hf
from mpl_toolkits.mplot3d import Axes3D
from SpectralClustering import SpectralClustering
import numpy as np
import seaborn as sns
from SSC import SSC

D = 2
N = 400
K = 2
trueLabels = np.append(np.ones([int(N/2), 1]), 2 * np.ones([int(N/2), 1])).reshape(N,1)

var = 0.001

# cluster 1
X1 = np.random.randn(D, int(N/2))
X1 = X1 / np.tile(np.sqrt(sum(X1**2)), (D,1))
X1 = X1 + np.tile(np.sqrt(var) * np.random.randn(1, int(N/2)), (D,1))

# cluster 2
X2 = np.random.randn(D, int(N/2))
X2 = 2*X2 / np.tile(np.sqrt(sum(X2**2)), (D,1))
X2 = X2 + np.tile(np.sqrt(var) * np.random.randn(1, int(N/2)), (D,1))

X = np.append(X1, X2, axis=1)

nNeighbors = 20
wtype = 'constant'
sigma = 1
W = hf.myKNN(X, nNeighbors, wtype, sigma)

#apply spectral clustering
C = SSC(X,800,800,800,200)
#print(np.linalg.norm(W - C))
estLabels, Y = SpectralClustering(C, K, 1)
scErrUn, UnestLabels_SP = hf.missRate(trueLabels, estLabels)
print(f"Spectral Clustering error (unnormalized): {scErrUn}")

# estLabels, Y = SpectralClustering(W, K, 1)
# scErr, estLabels_SP = hf.missRate(trueLabels, estLabels)
# print(f"Spectral Clustering error (normalized): {scErr}")
#
# #look at the data in the transformed space
# plt.figure(figsize=(6,6))
# plt.scatter(Y[0, 0:int(N/2)], Y[1, 0:int(N/2)], marker='X')
# plt.scatter(Y[0, int(N/2):N], Y[1, int(N/2):N], marker='X')
# plt.title("Transformed Data")
# plt.show()



#plotting


# fig = plt.figure()
# ax = Axes3D(fig)
#
# ax.scatter(X1[0,:],X1[1,:],X1[2,:],'r')
# ax.scatter(X2[0,:],X2[1,:],X2[2,:],'g')
# ax.scatter(X3[0,:],X3[1,:],X3[2,:],'b')
# plt.show()
