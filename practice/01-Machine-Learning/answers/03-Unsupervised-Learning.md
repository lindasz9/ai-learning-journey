# Unsupervised Learning - Test

1. What is unsupervised learning?
> A type of ML where the model gets unlabeled data and tries to find patterns, groupings, or structure in the data.

2. What are the main categories of unsupervised learning, and what do they mean?
> The main category of unsupervised learning is clustering, where we group similar datapoints based on features. The other is dimensionality reduction, where we reduce the number of features (dimensions) while preserving important information.

3. How do these clustering algorithms work? When do we use them? What are some strengths and weaknesses? What do the attached concepts mean?

k-Means Clustering
> A clustering algorithm, where we randomly assign k centroid, then attach the closest data points to them, then update the centroids to the mean of assigned points, then repeating this process we get the final clusters. Used for large datasets with known, roughly spherical clusters. It's simple, scalable, effective on spherical clusters. But it assumes circular clusters, and it's sensitive to outliers and noises.

- Elbow method
> It's a method for choosing the optimal k value. It calculates WCSS (Weighted Cluster Sum of Squares) for some k values, then looks for the "elbow" on the plot.

Hierarchical Clustering
> A clustering algorithm where we start with all of the points separate (agglomerative / bottom-up), then we group together the closest ones, and repeat this step. We can even start with all the points together (divisive / top-down), the split it in to 2 groups, and repeat. Used when we want to explore hierarchical relationships without knowing the value of k. It doesn't need predefined k and we can visualize the process on dendrogram. But it's computationally expensive and hard to scale to large datasets.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
> A clustering algorithm that groups close points together and marks ouliers. It classifies points as core, border or noise. Used for irregularly shaped clusters with noise and unknown cluster count. It's robust to noises, handles any shpae of cluster and doesn't need a k value. But it struggles with clusters of changing density, and it's sensitive to parameters.

- Density
> How close the data points are packed.

- Eps (ε)
> A parameter describing the maximum distance that 2 points can have and still be in the same cluster.

- minPts   
> A parameter describing the minimum number of points required to form a cluster.

4. How do these dimensionality reduction techniques work? When do we use them? What are some strengths and weaknesses?

PCA (Principal Component Analysis)
> A dimensionality reduction algorithm, that transforms data into new axes, combining features to capture the most variance possible. It's used when the data is linear. It's fast, and retains key patterns, but it can only capture linear relationships.

- Principal component  
> A new axis that maximizes variance.

t-SNE (t-distributed Stochastic Neighbor Embedding)
> A dimensionality reduction algorithm, that can visualize complex non-linear clusters in 2D or 3D paying attention to local structure. It's used for non-linear data when you want to see the clusters in complex high-dimensional data. It can capture complex data but can be slow for large datasets and results vary due to randomness.

UMAP (Uniform Manifold Approximation and Projection)
> A dimensionality reduction algorithm similar to t-SNE, but it preserves both local and global structure and works better for large datasets. We use it when t-SNE is slow. It's faster, more stable and better at preserving overall data, but it's harder to interpret.

5. What do these evaluation metrics mean?

Silhouette Score
> A clustering metric measuring how similar a point is to it's own cluster vs to other clusters.

Davies–Bouldin Index
> A clustering metric measuring the average similarity between clusters. The lower the better.

Calinski-Harabasz Index
> A clustering metric of the ratio of how far the clusters are from each other and how tight the clusters are. The higher the better (more separate clusters).

Explained Variance
> A dimensionality reduction metric showing the amount of information retained after reducing the dimensions.

6. What do these concepts mean?

Centroid
> The center (mean) of a cluster

Cluster
> A group of similar data points.

Dendrogram
> A tree-like diagram used to visualize the arrangement of clusters.

Dimensionality  
> The number of features in a dataset.

Embedding  
> The process of converting complex data into numerical vectors keeping their meaning and relationships.

Global structure
> The overall shape and arrangement of clusters across the entire dataset.

Local structure  
> The shape and arrangement of points in a cluster.

Noise  
> Irrelevant data that don't belong to any cluster.

Outlier  
> A data point that lies far away from other points.

Variance 
> The amount of spread in the data. 
