from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

s = 289963
np.random.seed(s)

def main():

    X, labels = make_circles(n_samples=500, noise=0.1, factor=.2)
    fig1, ax1 = plt.subplots()
    ax1.scatter(X[:, 0], X[:, 1])
    fig1.savefig('fig1.eps', format='eps')
    ax1.set_title('Raw data points')
    np.random.seed(s)
    kmeans = KMeans(n_clusters=2, random_state=s).fit(X)
    fig2, ax2 = plt.subplots()
    ax2.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    ax2.set_title('Clustered data using KMeans')
    fig2.savefig('fig2.eps', format='eps')
    np.random.seed(s)
    s_cluster = SpectralClustering(n_clusters = 2, eigen_solver='arpack',
            affinity="nearest_neighbors").fit_predict(X)
    fig3, ax3 = plt.subplots()
    ax3.scatter(X[:, 0], X[:, 1], c = s_cluster)
    ax3.set_title('Clustered data using spectral clustering')
    fig3.savefig('fig3.eps', format='eps')

    plt.show()
if __name__ == '__main__':
    main()