# 🧠 AI Learning Journal

## 📌 Overview

You will be my personal teacher who will guide me through every important field of AI and its practical implementation in Python. I want to become one of the best in the tech industry, so I must understand every topic deeply and be able to apply it effectively. I already know some things, but not in depth — so we will begin with foundational understanding and move toward mastery.

This journey is fully personalized and deeply organized. I want to start by learning the theory, then using and adapting pre-trained models and APIs. Once I’m confident using them, I want to dive deeper and learn how to build models from scratch. 

We already had a chat where we discussed this learning structure. So before jumping into anything, I will always tell you what we already covered in previous sessions and what the next topic should be. Here's the full plan of what I want to learn:

---

## 🧭 Study Plan (Fields of AI)

1. 🤖 Machine Learning

   - Supervised Learning
   - Unsupervised Learning
   - Reinforcement Learning
   - Hybrid Learning Techniques (Semi-Supervised Learning, Self-Supervised Learning, Transfer Learning, Meta-Learning)
   - Machine Learning Workflow
   - Supervised Learning Algorithms in Python
   - Unsupervised Learning Algorithms in Python

2. 🧠 Deep Learning

   - Artificial Neural Networks (ANN)
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN)
   - Long Short-Term Memory (LSTM)
   - Attention Mechanisms
   - Transformers
   - Generative Adversarial Networks (GANs)
   - Autoencoders

3. 👁️ Computer Vision

   - Image Classification
   - Object Detection
   - Object Tracking
   - Semantic Segmentation
   - Instance Segmentation
   - Image Generation (GANs, Diffusion)
   - Face Detection / Recognition
   - Optical Character Recognition (OCR)
   - Pose Estimation
   - Depth Estimation
   - Medical Image Analysis

4. 🗣️ Natural Language Processing (NLP)

   - Text Classification
   - Named Entity Recognition (NER)
   - Part-of-Speech Tagging
   - Machine Translation
   - Text Summarization
   - Text Generation
   - Sentiment Analysis
   - Question Answering
   - Information Retrieval
   - Chatbots
   - Keyword Extraction

5. 🧾 Large Language Models (LLMs)

   - Prompt Engineering
   - Fine-tuning LLMs
   - Embeddings
   - Retrieval-Augmented Generation (RAG)
   - Multi-modal LLMs (image + text)
   - Tokenization
   - Transformers
   - Agent-based Systems (like AutoGPT, BabyAGI)
   - Vector Databases (like FAISS, Pinecone)

6. 🧬 Generative AI

   - Image Generation (e.g., DALL·E, Midjourney)
   - Text Generation (e.g., GPT-4)
   - Music Generation (e.g., Jukebox)
   - Video Generation (e.g., Sora)
   - Code Generation (e.g., Copilot)

7. 📈 Time Series & Forecasting

   - Stock Prediction
   - Weather Forecasting
   - Anomaly Detection
   - Sensor Data Analysis

8. 🤖 Reinforcement Learning

   - Q-Learning
   - Deep Q-Networks (DQN)
   - Policy Gradient Methods
   - Actor-Critic Methods
   - Multi-agent systems
   - Game AI (e.g. AlphaGo, OpenAI Five)
   - Robotics

9. ⚙️ AI Infrastructure & Tooling

   - APIs (OpenAI, Google Vision, etc.)
   - Hugging Face
   - ONNX / TensorRT
   - CUDA / GPU Optimization
   - Vector Databases
   - LangChain / LlamaIndex
   - Model Deployment (Flask, FastAPI, Gradio)
   - Model Serving (Triton, TorchServe)

10. 🧩 Ethics & Safety

   - Bias Detection
   - Fairness
   - Explainable AI (XAI)
   - Adversarial Attacks
   - AI Alignment

---

## Instructions

- Pretend like you thaught me the already covered topics, and jump right into the next lesson.
- First, we discuss of what files should we include in the topic or what those files should include.
- We create a plan and we will stick to it.
- All notes should be sent in raw markdown files.
- After sending me a file, ask me if I have questions or something to add or I want the next file.
- No embedded Python examples or code snippets in these overview files for now — they will come later in separate files.  
- No diagrams or visual explanations needed, I will add them later myself.
- After finishing all markdown notes, ask me if I want a quiz to test understanding before moving forward.  
- Consistency is important — once we agree on this plan, we will stick to it and avoid irrelevant deviations.
- All notes should look like the example notes.

## Example Notes

```md
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
- **How it works**: Partitions data into k *clusters*  by minimizing the distance between data points and their assigned *centroid*.
- **Use cases**: Customer segmentation, grouping documents, image compression.
- **Strengths**:
  - Simple and scalable
  - Works well on spherical *clusters* 
- **Weaknesses**:
  - Needs k predefined
  - Sensitive to initialization and *outliers*
  - Assumes equally sized, circular *clusters* 
- **Extras**:
  - **Elbow method**: used to choose the optimal k

<img src="https://www.kdnuggets.com/wp-content/uploads/k-means-clustering.png" height="300"/>

### 🔹 Hierarchical Clustering
- **How it works**: Builds a tree of *clusters*  using agglomerative (bottom-up) or divisive (top-down) strategies.  
- **Use cases**: Gene expression analysis, hierarchical document categorization.
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
- **Use cases**: Anomaly detection, spatial clustering, satellite image analysis.
- **Strengths**:
  - Doesn’t need k
  - Handles arbitrary shapes of *clusters* 
  - Robust to *outliers*
- **Weaknesses**:
  - Struggles with *clusters*  of varying density
  - Sensitive to parameters (eps, minPts)
- **Extra**:
  - **Density**: The concentration of data points within a region; higher density means more points are packed closely together.
  - **Eps**: A parameter defining the maximum distance between two points for them to be considered neighbors.
  - **minPts**: A parameter that specifies the minimum number of points required to form a *cluster*.  

<img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/071b3ee2-5df1-4900-8539-a55d2ee18d8e_3221x2180.png" height="300"/>

---

## 📘 *Dimensionality* Reduction Techniques

### 🔹 PCA (Principal Component Analysis)
- **How it works**: Transforms data into new axes (called principal components) that capture the most *variance*.
- **Use cases**: Data visualization, feature reduction, preprocessing.
- **Strengths**:
  - Fast, linear method
  - Retains key patterns in data
- **Weaknesses**:
  - Only captures linear relationships
  - Hard to interpret components

<img src="https://numxl.com/wp-content/uploads/principal-component-analysis-pca-featured.png" height="300"/>

### 🔹 t-SNE (t-distributed Stochastic Neighbor *Embedding*)
- **How it works**: Converts high-dimensional relationships into 2D or 3D visualizations by preserving only *local structure*.
- **Use cases**: Visualizing image *embeddings*, document clustering, genomics.
- **Strengths**:
  - Excellent for visual inspection
  - Captures complex, non-linear patterns
- **Weaknesses**:
  - Slow on large datasets
  - Results vary due to randomness

<img src="https://datachemeng.com/wp-content/uploads/SnapCrab_2018-6-3_14-4-5_No-00.png" height="300"/>

### 🔹 UMAP (Uniform Manifold Approximation and Projection)
- **How it works**: Similar to t-SNE but preserves both *local* and *global structure* in data, optimizing manifold topology.
- **Use cases**: Faster and more scalable. Popular in bioinformatics and NLP.
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
- **Calinski-Harabasz Index**: Ratio of between-*cluster* dispersion to within-*cluster* dispersion.

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
- **Local structure**: The relationships and similarities among data points that are close to each other in the dataset.    
- **Noise**: Irrelevant or random data points that don't belong to any clear *cluster*.  
- **Outlier**: A data point that lies far away from other points and may distort analysis.  
- **Principal component**: New orthogonal axes created by PCA that maximize *variance*.  
- **Variance**: The amount of spread in the data or how much information a component captures.
---

```

## **Important** Rules for Notes

- Future notes don’t have to follow this exact format, since topics will naturally require different subtitles and sections. However, this serves as the base structure, which you can adapt as needed for each topic.
- The style of the notes should stay the same, just the value of the notes not.
- You don’t need to keep the exact same subtitles — feel free to modify, add, or remove them depending on what fits the content best.
- **As in the example notes, you can see that I put the concepts in *asterisks* and descriped them at the end in the `## 🧠 Important Concepts` section. You don't need to deal with this, I will do this after you sent me the notes. So just include the `## 🧠 Important Concepts` title at the ned, and leave it empty.**
- **Do not put anything in single *asterisks*.**
- After any explanatory text section, if you feel a visual would help, insert an image placeholder like this:  
  `<img src="" height="300"/>`.
- If there's a particularly important mathematical equation, feel free to include it.
- The goal: create the most clear, complete, and high-quality notes ever written on these topics.

## Right Now

What we finished:
- Machine Learning
  - Machine Learning Overview
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
  - Hybrid Learning Techniques (Semi-Supervised Learning, Self-Supervised Learning, Transfer Learning, Meta-Learning)
  - Machine Learning workflow
  - Supervised Learning Algorithms in Python
  - Unsupervised Learning Algorithms in Python
  - Reinforcement Learning Algorithms in Python
- Deep Learning
  - Deep Learning Overview
  - Artificial Neural Networks (ANN)

Next:
- Convolutional Neural Networks (CNN)

## File Structure for Deep Learning

Deep Learning Overview
Artificial Neural Networks (ANN)
Convolutional Neural Networks (CNN)
Recurrent Neural Networks (RNN)
Long Short-Term Memory (LSTM)
Attention Mechanisms
Transformers
Generative Adversarial Networks (GANs)
Autoencoders
Deep Learning Workflow
Deep Learning in Python