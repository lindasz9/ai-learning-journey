# Supervised Learning

## 🧠 What is Supervised Learning?

Supervised learning is a type of machine learning where the model is trained on a labeled dataset.  
Each data point includes both the input (features) and the correct output (label or target).

---

## 📂 Types of Supervised Learning Tasks

### 1. **Regression**
- Predicts **continuous numeric values**.
- **Real-world examples**: 
  - House prices
  - Temperature
  - Stock values

### 2. **Classification**
- Predicts **categorical labels** (classes).
- **Real-world examples**: 
  - Spam detection
  - Image classification
  - Tumor benign/malignant

---

## 📘 Regression Algorithms

### 🔹 Linear Regression
- **How it works**: Models a linear relationship (`y = wx + b`) between input features and the target.
- **When to use**: Simple problems with continuous target values and a linear relationship between inputs and output.
- **Strengths**:  
  - Easy to implement  
  - Easy to interpret  
  - Easy to visualize  
- **Weaknesses**:  
  - Fails with non-linear patterns  
  - Sensitive to outliers  

<img src="https://www.researchgate.net/publication/340271573/figure/fig3/AS:874657431437319@1585545990533/Linear-Regression-model-sample-illustration.ppm" height="300"/>

---

### 🔹 Ridge & Lasso Regression
- **How it works**: Add *regularization* terms to linear regression to penalize large *coefficients*.
- **When to use**: High-dimensional data or when multicollinearity is present.
- **Strengths**:  
  - Reduces overfitting  
  - Improves generalization  
- **Weaknesses**:  
  - Still assumes linear relationships  
- **Types**:  
  - **Lasso**: L1 penalty — can shrink some *coefficients* to zero (feature selection).  
  - **Ridge**: L2 penalty — shrinks *coefficients* but keeps all.  
  - **Elastic Net**: Combines L1 and L2 penalties — balances feature selection and coefficient shrinkage.  

<img src="https://images.datacamp.com/image/upload/v1648205672/image18_a3zz7y.png" height="300"/>

---

### 🔹 Support Vector Regression (SVR)
- **How it works**: Fits a function within a margin (*epsilon*) around the target. Only penalizes predictions outside that margin.
- **When to use**: Regression tasks where you want to control the tolerance for prediction error.
- **Strengths**:  
  - Effective in high-dimensional spaces  
  - Works with non-linear relationships using kernels  
- **Weaknesses**:  
  - Computationally expensive  
  - Sensitive to parameter tuning  

<img src="https://miro.medium.com/max/552/0*407C6bjGggCsN92U.png" height="300"/>

---

## 📙 Classification Algorithms

### 🔹 Logistic Regression
- **How it works**: Calculates probability using the *sigmoid function*, mapping predictions between 0 and 1. Classifies based on a threshold.
- **When to use**: Binary or multi-class classification with linearly separable classes.
- **Strengths**:  
  - Simple  
  - Interpretable  
  - Fast  
- **Weaknesses**:  
  - Assumes linear boundaries between classes  
  - Limited to simple decision boundaries  

<img src="https://almablog-media.s3.ap-south-1.amazonaws.com/image_15_ea63b72d9e.png" height="300"/>

---

### 🔹 k-Nearest Neighbors (kNN)
- **How it works**: Classifies a point based on the majority class among its k nearest neighbors in the feature space.
- **When to use**: Small datasets where relationships between points are intuitive.
- **Strengths**:  
  - No training phase  
  - Easy to understand  
  - Easy to implement  
- **Weaknesses**:  
  - Slow with large datasets  
  - Sensitive to irrelevant features  
  - Sensitive to feature scaling  
- **Extra**:  
  - **Cross-validation**: Evaluates model reliability and helps tune *k*.  
  - **Euclidean distance**: Measures distance between points to find neighbors.  

<img src="https://i0.wp.com/spotintelligence.com/wp-content/uploads/2023/08/k-nearest-neighbours-1024x576.webp?resize=1024%2C576&ssl=1" height="300"/>

---

### 🔹 Decision Trees
- **How it works**: Splits data into branches based on feature thresholds. Each node asks a question; leaves give predictions.
- **When to use**: Problems where interpretability is needed or features are both numerical and categorical.
- **Strengths**:  
  - Easy to interpret  
  - Handles both regression and classification  
- **Weaknesses**:  
  - Easily overfits without pruning or depth limitation  

<img src="https://venngage-wordpress.s3.amazonaws.com/uploads/2019/08/what-is-a-decision-tree-5.png" height="300"/>

---

### 🔹 Naive Bayes
- **How it works**: Uses *Bayes’ theorem* with the assumption that all features are conditionally independent given the class.
- **When to use**: Text classification problems, such as spam filtering or sentiment analysis.
- **Strengths**:  
  - Fast  
  - Scalable  
  - Works well with high-dimensional sparse data  
- **Weaknesses**:  
  - Assumes independence between features (rarely true in real life)  
- **Types**:  
  - **Gaussian NB**  
  - **Multinomial NB**  
  - **Bernoulli NB**  

<img src="https://databasecamp.de/wp-content/uploads/naive-bayes-overview.png" height="300"/>

---

### 🔹 Support Vector Machines (SVM)
- **How it works**: Finds the optimal *hyperplane* that separates classes with the maximum margin. Uses *kernel* trick for non-linear problems.
- **When to use**: Medium-sized datasets with clear class separation or high-dimensional feature spaces.
- **Strengths**:  
  - Effective in high dimensions  
  - Flexible via kernels  
- **Weaknesses**:  
  - Computationally intensive  
  - Requires careful tuning  
- **Types**:  
  - **Linear SVM**  
  - **Non-linear SVM**  

<img src="https://databasecamp.de/wp-content/uploads/svm.png" height="300"/>

---

## 🌲 Ensemble and Boosting Methods

### 🔹 Random Forest
- **How it works**: Builds multiple decision trees using random subsets of features and samples (*bagging*). Each tree gives a prediction; the final result is based on majority vote (classification) or average (regression).
- **When to use**: When you need a strong *baseline model* for tabular data, especially when interpretability and *overfitting* control are important.
- **Strengths**:  
  - Handles missing values  
  - Reduces overfitting  
  - Works with both numerical and categorical data  
- **Weaknesses**:  
  - Large models can be slow to predict  
  - Less interpretable than single decision trees  

<img src="https://miro.medium.com/v2/resize:fit:1200/1*jE1Cb1Dc_p9WEOPMkC95WQ.png" height="300"/>

---

### 🔹 Gradient Boosting
- **How it works**: Builds decision trees sequentially. Each new tree is trained to correct the residual errors of the previous one using *gradient descent* on a *loss function*.
- **When to use**: Complex structured data problems where accuracy is more important than interpretability or speed.
- **Strengths**:  
  - Extremely powerful and accurate  
  - Supports custom loss functions  
  - Handles missing values  
  - Supports parallelism  
- **Weaknesses**:  
  - Can overfit if not tuned properly  
  - More complex to configure and understand  
- **Types**:  
  - **XGBoost**: Adds regularization, early stopping, GPU support  
  - **LightGBM**: Faster for large datasets via histogram and leaf-wise growth  
  - **CatBoost**: Handles categorical features automatically, reduces prediction shift, good for heterogeneous data  

<img src="https://rohitgr7.github.io/content/images/2019/03/Screenshot-from-2019-03-27-23-09-47-1.png" height="300"/>

---

## 📊 Model Evaluation Metrics

### Regression:
- **MSE (Mean Squared Error)**: Average of squared errors
- **RMSE**: Square root of MSE
- **MAE**: Average of absolute errors
- **R² Score**: Proportion of variance explained by the model

### Classification:
- **Accuracy**: (Correct predictions) / (Total predictions)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC (Area Under the ROC Curve)**: Measures the ability of the classifier to distinguish between classes across all thresholds
- **Confusion Matrix**: Table comparing predicted vs actual labels

<img src="https://miro.medium.com/max/2560/1*mdtqR2kyElMd0cCGM4gtuw.jpeg" height="300"/>

---

## 🧠 Important Concepts

- **Bagging**: Short for bootstrap aggregating; a technique to improve model stability by training multiple models on random subsets of data.  
- **Baseline model**: A simple model used as a reference point to compare the performance of more complex models. It helps determine if a new model actually improves predictions.  
- **Bayes' theorem**: A formula that describes the probability of an event based on prior knowledge of conditions related to the event.  
- **Coefficient**: A numeric factor multiplying a feature in a model equation.  
- **Cross-validation**: A model evaluation method used to assess how the results of a statistical analysis will generalize to an independent data set. Often used to find the best hyperparameters such as k in kNN.  
- **Decision boundaries**: The hypersurfaces or curves that separate different classes in a classification problem. A model uses decision boundaries to assign labels to input data points.  
- **Epsilon**: A margin of tolerance in Support Vector Regression within which no penalty is given for errors.  
- **Gradient descent**: A method used to make a model better by slowly changing its parameters to reduce mistakes.
- **Hyperplane**: A flat, (n-1)-dimensional subspace that separates classes in classification problems, especially in Support Vector Machines (SVM).  
- **Histogram**: A graph that groups data into bins and shows how many data points fall into each, helping visualize the data’s distribution.  
- **Kernel**: A function that transforms data into a higher-dimensional space to make it easier to find a linear separation (linear/polynomial/RBF).
- **Loss Function**: A function measuring how far predictions are from actual values.  
- **Optimization**: The process of adjusting model parameters to minimize the loss function.  
- **Overfitting**: When a model fits the training data too closely, capturing noise instead of the underlying pattern, resulting in poor generalization.  
- **Pruning**: The process of removing parts of a decision tree that provide little predictive power to reduce complexity and prevent overfitting.  
- **Regularization**: Adding penalties to model parameters to prevent overfitting.  
- **Sigmoid function**: A function that maps any real-valued number into a value between 0 and 1, commonly used to model probabilities. 
