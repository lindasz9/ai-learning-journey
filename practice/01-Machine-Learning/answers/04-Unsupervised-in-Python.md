# Unsupervised Learning Algorithms – Code Test

1. How do you implement **k-Means Clustering**?
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhuette_score

X, y = make_blobs(n_samples=300, centers=3, random_state=42)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

print("Silhouette Score: ", silhouette_score(X, labels))
```

2. How do you implement **Hierarchical Clustering**?  
```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)
```

3. How do you implement **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**?
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)
```

4. How do you implement **Dimensionality Reduction Techniques**?
  
- PCA (Principal Component Analysis)  
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained Variance Ratio: ", pca.explained_variance_ratio_)
```

- t-SNE (t-distributed Stochastic Neighbor Embedding)  
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_tarnsform(X)

print("t-SNE Output Shape: ", X_tsne.shape)
```

- UMAP (Uniform Manifold Approximation and Projection)  
```python
import umap

reducer = umap.UMAP()
X_umap = reducer.fit_tarnsform(X)

print("UMAP Output Shape: ", X_umap.shape)
```
