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
9. 🚀 Deploy the Model

---

## 📥 Collect Data

The first step in any ML project is to gather data that represents the problem space.

**Where to get**:
- **Internal systems**: databases, logs, CRM, sensors.
- **Public datasets**: e.g. Kaggle, UCI Machine Learning Repository, government portals.
- **APIs**: e.g. financial data, weather data, social media platforms.
- **Web scraping**: extracting data from websites.
- **Manual labeling or crowdsourcing**: e.g. via Amazon Mechanical Turk.
- **Simulated data**: When real data is scarce or sensitive, generate synthetic data that mimics real distributions.

**Good data**:
- Representative of the real-world use case
- Diverse enough to capture edge cases
- Accurately labeled (in supervised learning)
- Clean, free from errors, duplicates, and missing values as much as possible
- Sufficient quantity, enough samples to train a reliable model

**Extra**:
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

**Integration**:  
- Merging data from multiple sources or tables  
- Resolving conflicts in data types or naming  
- Aligning formats (e.g. dates, currencies)
**Handling outliers**: Dealing with data points that deviate significantly from others.  
- **Clipping**: Limit the values of extreme outliers to a maximum/minimum threshold.  
- **Transformation**: Apply functions to reduce the impact of outliers (e.g. log).  
- **Removal**: Delete data points that are too far from the rest.  
- **Z-score method**: Identify outliers by checking how far values deviate from the mean in terms of standard deviations.

**Encoding**: The process of converting categorical variables into numerical format so they can be used by ML models.  
- **One-hot encoding**: Convert categorical values into binary vectors.  
- **Label encoding**: Assign each unique category an integer value.  
- **Ordinal encoding**: Similar to label encoding, but used when the order of categories matters.  
- **Frequency encoding**: Replace each category with its frequency in the dataset.  
- **Target encoding**: Replace each category with the mean of the target variable for that category.  

**Feature engineering**:  
- Creating new features from existing ones (e.g. extracting year from a date)  
- Combining features (e.g. total = quantity × price)  
- Removing irrelevant or redundant features  
- Converting timestamps into useful features (e.g. day of week, month)

**Transforming**: Changing the distribution of features to make them more suitable for modeling.  
- **Scaling**: The process of adjusting feature values to be on a similar scale.  
  - **Standardization**: Rescale features so they have mean 0 and standard deviation 1.  
  - **Normalization**: Rescale features to a [0, 1] range.  
- **Log transformation**: Reduce skew in numerical features.  
- **Box-Cox transformation**: Normalize data to make it more Gaussian.  

**Dimensionality reduction**:  
- **PCA (Principal Component Analysis)**: Reduce the number of features while retaining variance  
- **t-SNE / UMAP**: For visualization or simplifying complex datasets  
- Helps reduce overfitting and improve computation speed

---

## ✂️ Split Data into Training and Test Sets

To evaluate models fairly, we separate data into different sets:
- **Training set**: Used to train the model.
- **Validation set**: Used to tune hyperparameters and avoid overfitting.
- **Test set**: Held back until final evaluation.

**Extra**:
- **Sampling**: The process of selecting a subset of data from a larger dataset to train or evaluate a model.
  - **Stratified sampling**: Sampling that preserves the percentage of samples for each class.
- **Chronological splits**: Splitting time-ordered data in a way that preserves the sequence and prevents future data from leaking into the past.
- **Cross-validation**: A method for estimating model performance by splitting the dataset into several training/testing subsets.
- **Data leakage**: Occurs when information from outside the training dataset is used to create the model, leading to overoptimistic results.

<img src="https://algotrading101.com/learn/wp-content/uploads/2020/06/training-validation-test-data-set-1024x552.png" height="300"/>

---

## 🧠 Choose a Model

Choosing the right model depends on the type of task and data characteristics.

**Extra**:  
- **Baseline model**: A simple model used as a reference to evaluate the performance of more complex models.  
- **Model complexity**: Refers to how flexible a model is in fitting the training data. Simpler models are easier to interpret but may underfit.  
- **Task type**: The nature of the problem (classification, regression, clustering) guides which models are appropriate.  
- **Bias-variance trade-off**: Balancing underfitting and overfitting by choosing models with the right complexity.

---

## 🏋️ Train the Model

Model training means finding patterns in the training data.

- During training, the model adjusts parameters to minimize the *loss function*.
- The process tries to generalize, not memorize.

**Steps**: 
1. **Hyperparameter tuning**  
   - **Batch size**: Number of training samples processed before the model's parameters are updated.  
   - **Epochs**: Number of complete passes through the entire training dataset.  
   - **Learning rate**: Controls the size of the steps taken during optimization; affects speed and stability of convergence.  
   - **Optimizer**: Algorithm used to minimize the loss function (e.g. Gradient Descent, SGD, Adam).  
2. **Forward pass**: Input data is passed through the model to generate predictions.  
3. **Calculate loss**: Compare the model’s predictions with the true targets using a loss function.  
4. **Backward pass (Backpropagation)**: Compute gradients of the loss with respect to model parameters.  
5. **Optimization (e.g., Gradient Descent)**: Update model parameters using the gradients to minimize the loss.  
   - **Weight**: Parameters that influence input features. 
   - **Bias**: A value added to help the model make better predictions by shifting the output up or down.  
6. **Repeat**: Iterate the forward and backward passes for many epochs over the training data.  
7. **Validation**: Periodically evaluate the model on a validation set to monitor performance and prevent overfitting.    

Important **risks**:
- **Overfitting**: The model performs well on training data but poorly on new data.
- **Underfitting**: The model is too simple to capture patterns in the data.

Common **strategies**:
- **Early stopping**: Halt training when validation performance declines.
- **Regularization**: Penalize overly complex models to reduce *overfitting*.
- **Batch training**: Train on subsets of data to improve efficiency and generalization.

**Extra**:
- **Loss function**: A mathematical function that measures the error between predicted and true values.
- **Optimization**: The process of minimizing the *loss function* to improve the model's performance.
- **Learning rate**: A hyperparameter that controls how much the model's weights change in response to the error each time it is updated.

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
- **Calibration**: How well predicted probabilities reflect reality, important for probabilistic models.

---

## 🎛️ Tune Hyperparameters

Models often have hyperparameters that are not learned from data and need tuning.

**Hyperparameters**: Settings chosen by human before training that control model behavior.
- Learning rate
- Batch size
- Number of epochs
- Regularization strength (L1 or L2)
- For tree-based models: number of trees, max depth

Common methods:
- **Grid search**: Try all combinations from a predefined list of values.
- **Random search**: Try randomly chosen combinations of hyperparameters.
- **Bayesian optimization**: Use past evaluation results to model the relationship between hyperparameters and performance.
- **Cross-validation**: Avoid overfitting to the validation set by using multiple data splits.

---

## 👁️ Test on Unseen Data

Only after final tuning should the model be tested on the held-out test set.

If performance is disappointing:
- Check for data distribution mismatch (dissimilar training and test set).
- Consider collecting more representative data.

**Extra**:
- **Data snooping**: Using test data during development, which invalidates evaluation.
- **Generalization gap**: The difference in performance between training/validation and test sets.
