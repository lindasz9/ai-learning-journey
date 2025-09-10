# Supervised Learning - Test

1. What is supervised learning?
> Supervised learning is a category of ML where the model is trained on unlabeled data, meaning input-output pairs.

2. What are the types of supervised learning, and what do they mean?
> There are two types of supervised learning: regression, where the target/label is a continuous numeric value, and there's classification, where the target/label is a categorical value/a class.

3. How do these algorithms work? When do we use them? What are some strengths and weaknesses? What do the attached concepts mean?

Linear Regression
> A regression algorithm, that finds a linear pattern (y = wx + b) between the input features and the target, and then makes predictions based on this linear model. We use it for simple, regression problems where there is a linear relationship between inputs and output. It's simple and easy to implement and to visualize, but it's sensitive to outliers and only works on linear data.

Ridge Regression
> Another algorithm that adds regularization to linear regression by shrinking the coefficients, however, in this case, the coefficients are reduced but never shrink all the way to zero.

Lasso Regression
> An algorithms that adds regularization to linear regression in order to reduce large coefficients and avoid overfitting. Some coefficients can even reduce to zero. We use it when we want to perform feature selection or reduce model complexity, especially in high-dimensional data. It reduces overfitting, but still assumes linear relationship.

Elastic Net Regression
> A third regularization algorithm that combines both lasso and ridge regression penalties. It can both shrink coefficients and perform feature selection by allowing some coefficients to become zero while others are only reduced.

Support Vector Regression (SVR)
> Another algortihm to reduce overfitting in linear regression. This defines a margin (called epsilon) around the target values. The model only penalizes errors (predictions) that fall outside this margin.

- Epsilon
> The margin of tolerance around the predicted values where only outer errors are penalized.

Logistic Regression
> Despite its name, this is a classification algorithm. It’s similar to linear regression, as it uses the same formula, but instead of predicting a continuous value, it applies a sigmoid function to output probabilities, and the class with the highest probability is selected as the predicted target. It's useable when the classes are linearly separable. It's simple and fast, but fails on non-linear data.

k-Nearest Neighbors (kNN)
> A classification algorithm where the class of a new data point is determined by its k nearest neighbors in the feature space. It's used on simple, small datasets. It doesn’t require a real training phase and is easy to implement, but can be slow and sensitive to noise or feature scaling.

- Euclidean distance
> A type of distance metric that measures the straight-line distance between two points in a multi-dimensional space.

Decision Trees
> A classification and also a regression algorithm that uses a tree-like structure of nodes. Each internal node splits the data based on a feature, and this process continues down the branches until a leaf node is reached, which gives the final categorical prediction. We use it when the features are both numerical and categorical, and they contain interactions with the target. It's easy to interpret and can handle both regression and classification problems, but it's prone to overfitting.

Naive Bayes
> A classification algorithm based on Bayes' Theorem, assuming that all features are conditionally independent given the class label. We use it for text classification problems. It is fast and performs well with high-dimensional data, but it is only effective when the features are independent of each other.

Support Vector Machines (SVM)
> A classification algorithm that finds the optimal hyperplane to separate classes with the largest possible margin, maximizing the distance between the closest points of each class. We use it when classes can clearly be separated. It's effective with high dimensional data, but it's computationally expensive.

- Support vector
> A data point from a class that lies closest to the decision boundry.

4. How does these ensemble methods work? When do we use them? What are some strengts and weaknesses?

Random Forest
> A method that builds multiple decision trees using random subsets of the data. Each tree gives a prediction, the final result is based on majority votes (classification) or average (regression). We use it to improve accuracy compared to a single decision tree. It prevents overfitting, but can be slower.

Gradient Boosting
> This is also a method that uses multiple decision trees, however, it builds the trees sequentially, with each following tree trained to correct the errors of the previous one using a loss function and gradient descent. It is used when accuracy is more important than speed. This method is very powerful but also quite complex.

5. What do these evaluation metrics mean?

MSE
> Mean squared error

RMSE
> The square root of the mean squared error

MAE
> Mean absolute error

R² Score
> How much the target variable can be explained by the features.

Accuracy
> (TP + TN) / (P + N)

Precision
> TP / (TP + FP)

Recall
> TP / (TP + FN)

F1-Score
> 2 * Precision * Recall / (Precision + Recall) = 2 * TP / 2 * TP + FP + FN

AUC (Area Under the ROC Curve)
> Measures the ability of the model to distinguish between classes at all thresholds.

- ROC Curve
> Plots TPR (TP / (TP + FN)) and FPR (FP / (FP + TN)) at all thresholds.

Confusion Matrix
> A 2x2 table that compares TP, FP, FN and TN labels.

6. What do these concepts mean?

Bagging (bootstrap aggregating)
> The process of training multiple copies of a model on different bootstrap samples and aggregating their predictions to improve accuracy and reduce overfitting.

- Bootstrap samples
> Random subsets of the data sampled with replacement.

Baseline model  
> A simple model used for comparison, to check whether more complex models actually improve performance.

Bayes' theorem  
> A formula that updates the probability of an event based on prior knowledge and new evidence.

Coefficient  
> A numerical factor that multiplies a variable in a mathematical expression or term.

Cross-validation  
> A way of checking how well a ML model will generalize by splitting the data into multiple train/test parts, training on some, and validating on the rest, then averaging the results.

- k-fold
> A method where the dataset is split into k equal parts. The model is trained on k−1 parts and tested on the remaining part. This process repeats k times, and the average result tells us how good the model is.

Decision boundaries  
> The hypersurfaces or curves that separate different classes in a classification problem.

Feature space
> The multi-dimensional space where each dimension represents one feature of the data. Each data point is a single point in this space, based on its feature values.

Gradient descent  
> A method used to make a model better by slowly changing its parameters to reduce mistakes.

Hyperplane  
> A flat, (n-1)-dimensional subspace that separates classes in classification problems.

- Hypersurface
> A curved or nonlinear (n−1)-dimensional surface.

Histogram  
> A graph that groups data into bins and shows how many data points fall into each, helping visualize the data’s distribution.

Kernel  
> A function that transforms data into a higher-dimensional space to make it easier to find a linear separation (linear/polynomial/RBF).

Loss Function  
> A function measuring how far predictions are from actual values. 

Multicollarity
> When two or more features in a regression model are highly correlated, making it difficult to determine their individual effects on the target variable.

Optimization  
> The process of adjusting model parameters to minimize the loss function.

Overfitting  
> When a model fits the training data too closely, capturing noise instead of the underlying pattern, resulting in poor generalization.

Pruning  
> The process of removing unnecessary parts of a model or search space to simplify it and reduce overfitting.

Regularization  
> Adding penalties to model parameters to prevent overfitting.  

Sigmoid function  
> A function that maps any real-valued number into a value between 0 and 1, commonly used to model probabilities. 
