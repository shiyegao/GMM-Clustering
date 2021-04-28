import numpy as np
import sklearn.mixture import GaussianMixture



def Gaussian_Mixture(data, centroids, assignment=None, iter=15, axes=None):
    for i in range(iter):
        responsibilities = E_step(X, means, covs, weights)
        assignments = responsibilities.argmax(axis=1)


gmm = GaussianMixture(n_components=K, covariance_type='full').fit(X)
y_gmm = gmm.predict(X)
_, ax = plt.subplots(figsize = (8,8))
scatter(X, y_gmm, ax=ax)



