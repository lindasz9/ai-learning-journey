# Unsupervised Learning

## 🧠 What is Unsupervised Learning?

Unsupervised learning is a type of machine learning where models are trained on **unlabeled data**.  
The algorithm learns to find patterns, structures, or groupings in the data without explicit instructions.

---

## 📂 Main Categories of Unsupervised Learning

### 1. **Clustering**
- **What it is**: Clustering is the task of grouping similar data points together based on their features.  
- **Real-world examples**:
  - Customer segmentation in marketing
  - Grouping news articles by topic
  - Detecting fraudulent behavior
  - Recommending content based on user similarity

### 2. ***Dimensionality* Reduction**
- **What it is**: *Dimensionality* reduction reduces the number of features (or dimensions) while preserving important information in the data.
- **Real-world examples**:
  - Visualizing high-dimensional data in 2D/3D
  - Noise reduction in images
  - Compressing large datasets
  - Speeding up training for other ML models

---

## 📘 Clustering Algorithms

### 🔹 k-Means Clustering
- **How it works**: Partitions data into k *clusters* by minimizing the distance between data points and their assigned *centroid*.
- **Use cases**: Used for large datasets with known, roughly spherical clusters.
- **Strengths**:
  - Simple and scalable
  - Works well on spherical *clusters* 
- **Weaknesses**:
  - Needs k predefined
  - Sensitive to initialization and *outliers*
  - Assumes equally sized, circular *clusters* 
- **Extras**:
  - **Elbow method**: Used to choose the optimal k by calculating a formula for some k values, then looking for the "elbow" on the plot.

<img src="https://www.kdnuggets.com/wp-content/uploads/k-means-clustering.png" height="300"/>

### 🔹 Hierarchical Clustering
- **How it works**: Builds a tree of *clusters*  using agglomerative (bottom-up) or divisive (top-down) strategies.  
- **Use cases**: Used when we want to explore hierarchical relationships without knowing the value of k.
- **Strengths**:
  - Doesn’t need to predefine k
  - Results in a *dendrogram* showing *cluster* hierarchy
- **Weaknesses**:
  - Computationally expensive
  - Hard to scale to large datasets
- **Types**:
  - Agglomerative (most common)
  - Divisive

<img src="https://www.datanovia.com/en/wp-content/uploads/dn-tutorials/003-hierarchical-clustering-in-r/figures/005-visualizing-dendrograms-cutree-1.png" height="300"/>

### 🔹 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- **How it works**: Groups points that are closely packed together and marks isolated points as *noise*. Classifies points as core, border, or *noise*.
- **Use cases**: Used for irregularly shaped clusters with noise and unknown cluster count.
- **Strengths**:
  - Doesn’t need k
  - Handles arbitrary shapes of *clusters* 
  - Robust to *outliers*
- **Weaknesses**:
  - Struggles with *clusters* of varying density
  - Sensitive to parameters (eps, minPts)
- **Extra**:
  - **Density**: The concentration of data points within a region; higher density means more points are packed closely together.
  - **Eps (ε)**: A parameter defining the maximum distance between two points for them to be considered neighbors.
  - **minPts**: A parameter that specifies the minimum number of points required to form a *cluster*.  

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/071b3ee2-5df1-4900-8539-a55d2ee18d8e_3221x2180.png" height="300"/>

---

## 📘 *Dimensionality* Reduction Techniques

### 🔹 PCA (Principal Component Analysis)
- **How it works**: Transforms data into new axes (called principal components) that capture the most *variance*.
- **Use cases**: Used when the data is linear.
- **Strengths**:
  - Fast, linear method
  - Retains key patterns in data
- **Weaknesses**:
  - Only captures linear relationships
  - Hard to interpret components
- **Extra**:
  - - **Principal component**: New orthogonal axes that maximize *variance*.  

<img src="https://numxl.com/wp-content/uploads/principal-component-analysis-pca-featured.png" height="300"/>

### 🔹 t-SNE (t-distributed Stochastic Neighbor *Embedding*)
- **How it works**: Converts high-dimensional relationships into 2D or 3D visualizations by preserving only *local structure*.
- **Use cases**: Used when the data is non-linear and we wanat to see the clusters in high-dimensional data.
- **Strengths**:
  - Excellent for visual inspection
  - Captures complex, non-linear patterns
- **Weaknesses**:
  - Slow on large datasets
  - Results vary due to randomness

<img src="https://datachemeng.com/wp-content/uploads/SnapCrab_2018-6-3_14-4-5_No-00.png" height="300"/>

### 🔹 UMAP (Uniform Manifold Approximation and Projection)
- **How it works**: Similar to t-SNE but preserves both *local* and *global structure* in data, optimizing manifold topology.
- **Use cases**: Used for large, non-linear datasets where t-SNE is slow.
- **Strengths**:
  - Faster and more stable than t-SNE
  - Better at preserving overall data structure
- **Weaknesses**:
  - Harder to interpret

<img src="https://miro.medium.com/v2/resize:fit:1200/1*fGQImmija7kepddB7SFaGA.jpeg" height="300"/>

---

## 📊 Model Evaluation

### Clustering Metrics:
- **Silhouette Score**: Measures how similar a point is to its own *cluster* vs other *clusters* .
- **Davies–Bouldin Index**: Measures average similarity between *clusters*  (lower is better).
- **Calinski-Harabasz Index**: Ratio of between-*cluster* dispersion (how far clusters are from each other) to within-*cluster* dispersion (how tigth the clusters are) (higehr is better, it means that the clusters are more separate).

### *Dimensionality* Reduction:
- **Explained *Variance***: Shows how much information is retained in fewer dimensions.

---

## 🧠 Important Concepts

- **Centroid**: The center of a *cluster* (mean of all data points in that *cluster*).  
- **Cluster**: A group of data points that are similar to each other.  
- **Dendrogram**: A tree diagram used to visualize the arrangement of the *clusters*.
- **Dimensionality**: The number of features or attributes in a dataset.  
- **Embedding**: A process of converting complex data (like words, images, or nodes) into numerical vectors that capture their meaning or relationships.
- **Global structure**: The overall shape and arrangement of data clusters or patterns across the entire dataset.  
- **Local structure**: The relationships and similarities among data points in a cluster.    
- **Noise**: Irrelevant or random data points that don't belong to any clear *cluster*.  
- **Outlier**: A data point that lies far away from other points and may distort analysis.  
- **Variance**: The amount of spread in the data or how much information a component captures.
