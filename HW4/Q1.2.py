import numpy as np
from scipy.spatial.distance import cdist

mu1 = np.array([[5.3],
                [3.5]])
mu2 = np.array([[5.1],
                [4.2]])
D = np.array([[5.5, 3.1],
              [5.1, 4.8],
              [6.3, 3.0],
              [5.5, 4.4],
              [6.8, 3.5]])

# first iteration in K-means
d1 = cdist(D, mu1.T)
d2 = cdist(D, mu2.T)
c1 = (d1 < d2).flatten()
c2 = (d2 < d1).flatten()

mu1_updated = np.mean(D[c1], axis=0, keepdims=True).T
mu2_updated = np.mean(D[c2], axis=0, keepdims=True).T

print(mu1_updated)
print(mu2_updated)

d1_updated = cdist(D, mu1_updated.T)
d2_updated = cdist(D, mu2_updated.T)
c1_updated = (d1_updated < d2_updated).flatten()
c2_updated = (d2_updated < d1_updated).flatten()

print(np.sum(c1_updated))
print(np.sum(c2_updated))
