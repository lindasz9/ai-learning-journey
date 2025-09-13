# Machine Learning Workflow - Test

1. What are the steps in the ML workflow?
> First, we collect the data, then we clean it and preprocess it, then we split it into train and etst sets. We choose the appropriate model, we train it, we evaluate it, then we fine-tune the hyperparameters, and finally, we test it on unseen data.

2. What does "collect data" mean? What do the attached concepts mean?

Where can we get data?
> From internal systems (databases, sensors), public datasets (Kaggle, UCI), APIs (financial, weather, social media), web scraping (from websites), manual labeling or crowdsourcing (Amazon Mechanical Turk), and simulated data (synthetic data when real data is scarce or sensitive).

What do we consider "good" data?
> Good data means that it represents the real-world use case, it captures edge cases, it's clean, no errors, no duplicates or missing values, and there's enough samples.

Crowdsourcing
> Getting a large group of people to complete small tasks, like labeling data, to collectively build a dataset.

Feature
> An indicidual measurable property or characteristic of the data used as input for the model.

Target (Label)
> The output variable that the model is trying to predict.

Instance (Sample, Example)
> A single data point consisting of features and (optionally) a target.

3. What does "clean and preprocess data" mean? What do the attached concepts mean?

How do we clean the data?
> We fix or remove missing values using imputation, we remove any duplicates or corrupted entries, we correct mislabeled or inconsistent data, and we handle outliers.

- Imputation
> The process of filling in missing values in the dataset.

- Mean/median imputation
> Replacing missing values with the mean/median of the column.

- KNN imputation
> Replacing missing values with the average of nearest neighbours.

- Handling outliers
> Dealing with data points taht dveiate significally from others.

- Clipping
> Limit the values of extreme outliers to a maximum/minimum threshold.

- Removal
> Delete data points that are too far away from the rest.

- Z-score method
> Identify outliers by checking how far values deviate from the mean in terms of standard deviations.

What's data integration?
> We merge data from multiple sources or tables, we resolve conflicts in data types or naming, and we align formats.

What's encoding?
> It's the process of converting categorical variables into numerical format so they can be used by ML models.

- One-hot encoding
> Convert categorical values into binary vectors, where each position represents a category and only the position for the current category is 1, all others are 0.

- Label encoding
> Assign each unique category an integer value.

- Ordinal encoding
> Similar to label encoding, but used when the order of categories matters.

- Frequency encoding
> Replace each category with its frequency in the dataset.  

- Target encoding
> Replace each category with the mean of the target variable for that category.  

What's feature engineering?
> It's creating new features from existing ones, or combining features, or even removing irrelevant ones.

What's data transformation?
> The process of applying functions to data to change its scale, distribution, or format, in order to make it easier for the model to learn. We use scaling to standardize or normalize.

- Scaling
> The process of adjusting feature values to be on a similar scale.  

- Standarization
> Rescale features so they have mean 0 and standard deviation 1.  

- Normalization
> Rescale features to a [0, 1] range.  

- Log transformation
> Reduce skew in numerical features by the logarithm function.  

- Box-Cox transformation
> Normalize data to make it more Gaussian (bell-shaped, symmetric).  

How do we reduce dimensionality?
> We can use PCA if we want to retain variance or t-SNE or UMAP for visualization or simplifying complex datasets. This helps reduce overfitting, improve computation speed and visualizing.

4. What does "split data into training and test sets" mean? What do the attached concepts mean?
> To evaluate models fairly, we separate data into train, test, and sometimes validation sets.

Training set
> Data the model learns from.

Validation set
> Data used to tune parameters and check performance during training.

Test set
> Data used for the final evaluation after training.

Sampling
> The process of selecting a subset of data from a larger dataset to train or evaluate a model.

- Satisfied sampling
> Selecting a subset of data that meets specific requirements, instead of choosing completely at random.

Chronological splits
> Splitting time-ordered data in a way that preserves the sequence and prevents future data from leaking into the past.

Cross-validation
> A method for estimating model performance by splitting the training data into several subsets, training on some folds and validating on the others.

Data leakage
> Using information in training that wouldn’t be available in real predictions.

5. What does "choose a model" mean? What do the attached concepts mean?

Baseline model
> A simple model used as a reference to evaluate the performance of more complex models. 

Model complexity
> How complicated a model is in terms of its capacity to fit data.

Task type
> The nature of the problem (classification, regression, clustering) guides which models are appropriate.

Bias-variance tarde-off
> Balancing underfitting and overfitting by choosing models with the right complexity.

6. What does "train the model" mean? What do the attached concepts mean?

What are the steps?

- Hyperparameter tuning
> The process of finding the best settings for a model’s parameters set before training.

- Forward pass
> Input data is passed through the model to generate predictions.  

- Calculate loss
> Compare the model’s predictions with the true targets using a loss function. 

- Baskward pass (Backpropagation)
> The method for calculating how much each weight contributed to the error (calculating gradients).

- Optimization
> Update model parameters using the gradients to minimize the loss.  

- Validation
> The process of testing the model on a separate dataset during training to check performance.

What are the risks?
> The model can overfit the data meaning that the model performs too well on training data, but poorly on new data. The otehr risk is underfitting, when a model is too simple to capture patterns in the data.

What are common strategies?

- Early stopping
> Stops training when validation performance declines.

- Regularization
> Penalize overly complex models to reduce overfitting.

- Batch training
> Train on subsets of data to improve efficiency and generalization.

Batch size
> Number of training samples processed before the model's parameters are updated. 

Bias
> A value added to help the model make better predictions by shifting the output up or down. 

Epochs
> Number of complete passes through the entire training dataset. 

Loss function
> A mathematical function that measures the error between predicted and true values.

Learning rate
> A hyperparameter that controls the size of the steps when updating model weights.

Optimizer
> An algorithm used to minimize the loss function by updating parameters (e.g. Gradient Descent, SGD, Adam).  

Weight
> Parameters that influence input features. 

7. What does "evaluate the model" mean? What do the attached concepts mean?

What are some evaluation metrics?
> For regression problems, there are MSE, RMSE, MAE and R² score, and for classification problems, there are accuracy, precision, recall, F1 score and AUC.

What are some visualization tools?

- Confusion matrix
> A table showing correct vs incorrect predictions across classes.  

- ROC curve
> Plots true positive rate vs false positive rate at different classification thresholds. 

- PR curve
> Plots precision vs recall at different thresholds, especially useful for imbalanced datasets.

Calibration
> How well predicted probabilities reflect reality (how accurate the predicted probabilities are).

8. What does "tune hyperparameters" mean? What do the attached concepts mean?

What are some hyperparameters?
> Some hyperparameters are for example batch size, number of epochs, learning rate, regularization strength and for tree-based models there's more: number of trees, max depth.

What are some common methods for finding optimal hyperparameters?

- Grid search
> Try all combinations from a predefined list of values.

- Random search
> Try randomly chosen combinations of hyperparameters.

- Bayesian optimization
> Use past evaluation results to model the relationship between hyperparameters and performance.

- Cross-validation
> A method to evaluate model performance using multiple train/validation splits to prevent overfitting.

9. What does "test on useen data" mean? What do the attached concepts mean?
> The held-out test set should be first seen only after final tuning.

Data snooping
> Using information from the test set (or future data) during model training, which can lead to overly optimistic results.

Generalization gap
> The difference in performance between training/validation and test sets.
