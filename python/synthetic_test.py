from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from SSC import SSC
#Generating synthetic data
centers = [(-10, -6), (7, 7)]
cluster_std = [0.8, 1]

X, y = make_blobs(n_samples=100, cluster_std=cluster_std,
                centers=centers, n_features=2, random_state=1)

C = SSC(X.T,0.1,0.1,0.1,1)
print(C.shape)
