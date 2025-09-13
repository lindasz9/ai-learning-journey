# Machine Learning Workflow

The machine learning workflow is a step-by-step process used to develop models that can make predictions or decisions based on data. Each step builds upon the previous one, forming a pipeline from raw data to deployed system.

## 🧭 Steps

This workflow includes the following steps:

1. 📥 Collect Data  
2. 🧹 Clean and Preprocess Data  
3. ✂️ Split Data into Training and Test Sets  
4. 🧠 Choose a Model  
5. 🏋️ Train the Model  
6. 🧪 Evaluate the Model  
7. 🎛️ Tune Hyperparameters  
8. 👁️ Test on Unseen Data

---

## 📥 Collect Data

The first step in any ML project is to gather data that represents the problem space.

**Where to get**:
- **Internal systems**: Databases, sensors.
- **Public datasets**: e.g. Kaggle, UCI Machine Learning Repository.
- **APIs**: e.g. financial data, weather data, social media platforms.
- **Web scraping**: Extracting data from websites.
- **Manual labeling or *crowdsourcing***: Crowdsourcing with Amazon Mechanical Turk.
- **Simulated data**: When real data is scarce or sensitive, generate synthetic data that mimics real distributions.

**Good data**:
- Representative of the real-world use case
- Diverse enough to capture edge cases
- Clean, free from errors, duplicates, and missing values
- Sufficient quantity, enough samples to train a reliable model

**Extra**:
- **Crowdsourcing**: Getting a large group of people to complete small tasks, like labeling data, to collectively build a dataset.
- **Feature**: An individual measurable property or characteristic of the data used as input for the model.
- **Target (Label)**: The output variable the model is trying to predict.
- **Instance (Sample, Example)**: A single data point consisting of features and (optionally) a target.

---

## 🧹 Clean and Preprocess Data

Raw data is rarely ready for modeling. This step ensures it’s clean and consistent.

**Cleaning**:  
- Fixing or removing missing values  
  - **Imputation**: The process of filling in missing values in the dataset.  
    - **Mean/Median imputation**: Replace missing values with the column mean or median.  
    - **KNN imputation**: Use the average of nearest neighbors to estimate missing values.  
- Removing duplicates or corrupted entries  
- Correcting mislabeled or inconsistent data  
- **Handling outliers**: Dealing with data points that deviate significantly from others.  
  - **Clipping**: Limit the values of extreme outliers to a maximum/minimum threshold. 
  - **Removal**: Delete data points that are too far from the rest.  
  - **Z-score method**: Identify outliers by checking how far values deviate from the mean in terms of standard deviations.

**Integration**:  
- Merging data from multiple sources or tables  
- Resolving conflicts in data types or naming  
- Aligning formats (e.g. dates) 

**Encoding**: The process of converting categorical variables into numerical format so they can be used by ML models.  
- **One-hot encoding**: Convert categorical values into binary vectors, where each position represents a category and only the position for the current category is 1, all others are 0. 
- **Label encoding**: Assign each unique category an integer value.  
- **Ordinal encoding**: Similar to label encoding, but used when the order of categories matters.  
- **Frequency encoding**: Replace each category with its frequency in the dataset.  
- **Target encoding**: Replace each category with the mean of the target variable for that category.  

**Feature engineering**:  
- Creating new features from existing ones (e.g. extracting year from a date)  
- Combining features (e.g. total = quantity × price)  
- Removing irrelevant or redundant features  

**Transformation**: The process of applying functions to data to change its scale, distribution, or format, in order to make it easier for the model to learn.  
- **Scaling**: The process of adjusting feature values to be on a similar scale.  
  - **Standardization**: Rescale features so they have mean 0 and standard deviation 1.  
  - **Normalization**: Rescale features to a [0, 1] range.  
- **Log transformation**: Reduce skew in numerical features by the logarithm function.  
- **Box-Cox transformation**: Normalize data to make it more Gaussian (bell-shaped, symmetric).  

**Dimensionality reduction**:  
- **PCA (Principal Component Analysis)**: Reduce the number of features while retaining variance.
- **t-SNE / UMAP**: For visualization or simplifying complex datasets  
- Helps reduce overfitting and improve computation speed.

---

## ✂️ Split Data into Training and Test Sets

To evaluate models fairly, we separate data into different sets:
- **Training set**: Data the model learns from.
- **Validation set**: Data used to tune parameters and check performance during training.
- **Test set**: Data used for the final evaluation after training.

**Extra**:
- **Sampling**: The process of selecting a subset of data from a larger dataset to train or evaluate a model.
  - **Stratified sampling**: Selecting a subset of data that meets specific requirements, instead of choosing completely at random.
- **Chronological splits**: Splitting time-ordered data in a way that preserves the sequence and prevents future data from leaking into the past.
- **Cross-validation**: A method for estimating model performance by splitting the dataset into several training/testing subsets.
- **Data leakage**: Using information in training that wouldn’t be available in real predictions.

<img src="https://algotrading101.com/learn/wp-content/uploads/2020/06/training-validation-test-data-set-1024x552.png" height="300"/>

---

## 🧠 Choose a Model

Choosing the right model depends on the type of task and data characteristics.

**Extra**:  
- **Baseline model**: A simple model used as a reference to evaluate the performance of more complex models.  
- **Model complexity**: How complicated a model is in terms of its capacity to fit data.  
- **Task type**: The nature of the problem (classification, regression, clustering) guides which models are appropriate.  
- **Bias-variance trade-off**: Balancing underfitting and overfitting by choosing models with the right complexity.

---

## 🏋️ Train the Model

Model training means finding patterns in the training data.

- During training, the model adjusts parameters to minimize the *loss function*.
- The process tries to generalize, not memorize.

**Steps**: 
1. **Hyperparameter tuning**: The process of finding the best settings for a model’s parameters set before training.
2. **Forward pass**: Input data is passed through the model to generate predictions.  
3. **Calculate loss**: Compare the model’s predictions with the true targets using a loss function.  
4. **Backward pass (Backpropagation)**: A method used in training neural networks to update the model’s weights by calculating how much each weight contributed to the error and adjusting them to reduce that error.
5. **Optimization (e.g., Gradient Descent)**: Update model parameters using the gradients to minimize the loss.  
6. **Validation**: The process of testing the model on a separate dataset during training to check performance.  

Important **risks**:
- **Overfitting**: The model performs well on training data but poorly on new data.
- **Underfitting**: The model is too simple to capture patterns in the data.

Common **strategies**:
- **Early stopping**: Stops training when validation performance declines.
- **Regularization**: Penalize overly complex models to reduce *overfitting*.
- **Batch training**: Train on subsets of data to improve efficiency and generalization.

**Extra**:
- **Batch size**: Number of training samples processed before the model's parameters are updated.  
- **Bias**: A value added to help the model make better predictions by shifting the output up or down. 
- **Epochs**: Number of complete passes through the entire training dataset. 
- **Loss function**: A mathematical function that measures the error between predicted and true values.
- **Learning rate**: A hyperparameter that controls the size of the steps when updating model weights.
- **Optimizer**: An algorithm used to minimize the loss function by updating parameters (e.g. Gradient Descent, SGD, Adam).  
- **Weight**: Parameters that influence input features.  

---

## 🧪 Evaluate the Model

After training, evaluate model performance on validation or test data.

**Evaluation metrics**:  
- Regression: MSE, RMSE, MAE, R², etc.  
- Classification: Accuracy, precision, recall, F1 score, AUC (Area Under the ROC Curve), etc.

**Visual tools**:  
- **Confusion matrix**: A table showing correct vs incorrect predictions across classes.  
- **ROC curve**: Plots true positive rate vs false positive rate at different classification thresholds.  
- **PR curve**: Plots precision vs recall at different thresholds, especially useful for imbalanced datasets.

**Extra**:  
- **Calibration**: How well predicted probabilities reflect reality (how accurate the predicted probabilities are).

---

## 🎛️ Tune Hyperparameters

Models often have hyperparameters that are not learned from data and need tuning.

**Hyperparameters**: Settings chosen by human before training that control model behavior.
- Learning rate
- Batch size
- Number of epochs
- Regularization strength (L1 or L2)
- For tree-based models: number of trees, max depth

Common methods for finding optimal hyperparameters:
- **Grid search**: Try all combinations from a predefined list of values.
- **Random search**: Try randomly chosen combinations of hyperparameters.
- **Bayesian optimization**: Use past evaluation results to model the relationship between hyperparameters and performance.
- **Cross-validation**: A method to evaluate model performance using multiple train/validation splits to prevent overfitting.

---

## 👁️ Test on Unseen Data

Only after final tuning should the model be tested on the held-out test set.

If performance is disappointing:
- Check for data distribution mismatch (dissimilar training and test set).
- Consider collecting more representative data.

**Extra**:
- **Data snooping**: Using test data during development, which invalidates evaluation.
- **Generalization gap**: The difference in performance between training/validation and test sets.
