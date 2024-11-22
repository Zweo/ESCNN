# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:38:40 2022

@author: User01
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.io as io
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = io.loadmat('data/SC/feature2.mat')
x = data['feature']
numel = len(x)
comp = 3
x = np.array(x)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
pca = PCA(n_components=comp)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(sum(pca.explained_variance_ratio_))
nn = NearestNeighbors(n_neighbors=10, algorithm='auto')
nn.fit(X_pca)
distances, indices = nn.kneighbors(X_pca)

max_score = 0
plt.figure(figsize=(5, 5))
best_cluster = 2
for best_cluster in range(2, 6):
    kmeans_b = KMeans(n_clusters=best_cluster)
    print(best_cluster)
    kmeans_b.fit(X_pca)
    y = kmeans_b.labels_
    centers = kmeans_b.cluster_centers_
    tsne = TSNE(n_components=2, random_state=0)  # 降维到2D
    xx = np.vstack([X_pca, centers])
    xx = tsne.fit_transform(xx)
    plt.subplot(2, 2, best_cluster - 1)
    plt.scatter(xx[:-best_cluster, 0],
                xx[:-best_cluster, 1],
                c=y,
                cmap='viridis',
                s=50,
                edgecolor='k')

    plt.scatter(xx[-best_cluster:, 0],
                xx[-best_cluster:, 1],
                c='black',
                s=200,
                alpha=0.5,
                edgecolor='k',
                marker='x')
    plt.title(f"subgroups = {best_cluster}")
    plt.xticks([])
    plt.yticks([])
plt.show()
