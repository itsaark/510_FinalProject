from sklearn.datasets.samples_generator import make_blobs

#Generating synthetic data
centers = [(-10, -6), (7, 7)]
cluster_std = [0.8, 1]

X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
