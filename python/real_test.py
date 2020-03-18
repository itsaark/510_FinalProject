from sklearn.datasets.samples_generator import make_blobs
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from python.SSC import SSC

face_data = sio.loadmat('YaleBCrop025.mat')

# dataset has 38 individuals, each with 64 images (under different lighting)
# images are 48 x 42 (or 2016 as a vector)
I = face_data['I']  # with 2d images
Y = face_data['Y']  # with flattened images
m, n, d = Y.shape

#plotting to make sure data's being read in correctly
# fig = plt.figure()
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(I[:,:,0,0], cmap='gray')
# plt.show()

#extract desired number of face-groups
nfaces = 2
nfaces_data = Y[:, :, :nfaces].reshape(m, nfaces * n).T

#append labels for each face-group
labels = np.repeat([i for i in range(nfaces)], n).reshape((nfaces * n, 1))
nfaces_data = np.append(nfaces_data, labels, axis=1)

#shuffle the data (is this necessary?)
np.random.shuffle(nfaces_data)

X = nfaces_data[:, :m]
y = nfaces_data[:, -1:]

C = SSC(X.T, 0.1, 0.1, 0.1, 1)
print(C.shape)
print()