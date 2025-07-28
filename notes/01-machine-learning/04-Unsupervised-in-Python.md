# Unsupervised Learning Algorithms in Python

## Algorithms

📘 **Clustering Algorithms**
- k-Means Clustering
- Hierarchical Clustering
  - Agglomerative
  - Divisive
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

📙 **Dimensionality Reduction Techniques**
- PCA (Principal Component Analysis)
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

---

## 📘 Clustering Algorithms

### 🔹 k-Means Clustering
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

print("Silhouette Score:", silhouette_score(X, labels))
```

### 🔹 Hierarchical Clustering

#### Agglomerative
```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

print("Labels:", labels)
```

#### Divisive
*Note: There is no direct divisive clustering in scikit-learn. Typically implemented using custom algorithms or via dendrograms.*

### 🔹 DBSCAN
```python
from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

print("Labels:", labels)
```

---

## 📙 Dimensionality Reduction Techniques

### 🔹 PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```

### 🔹 t-SNE
```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

print("t-SNE Output Shape:", X_tsne.shape)
```

### 🔹 UMAP
```python
import umap

reducer = umap.UMAP()
X_umap = reducer.fit_transform(X)

print("UMAP Output Shape:", X_umap.shape)
```
