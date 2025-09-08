# Supervised Learning Algorithms in Python - Test

1. How do you implement **Linear Regression**?
```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
```

2. How do you implement **Ridge, Lasso, and Elastic Net Regression**?
  
- Ridge Regression  
```python
from sklearn.linear_model import Ridge

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

- Lasso Regression  
```python
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
```

- Elastic Net  
```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
```

3. How do you implement **Support Vector Regression (SVR)**?
```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("MSE: ", mean_squared_error(y_test, y_pred))
```

4. How do you implement **Logistic Regression**?
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
```

5. How do you implement **k-Nearest Neighbors (kNN)**?
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNearestClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

6. How do you implement **Decision Trees**?
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
```

7. How do you implement **Naive Bayes**?
  
- GaussianNB  
```python
from sklearn.niave_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

- MultinomialNB  
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

texts = ["spam email", "important project", "buy cheap", "urgent money offer"]
labels = [1, 0, 1, 1]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinumialNB()
model.fit(X, labels)
pred = model.predict(X)

print("Predictions: ", pred)
```

- BernoulliNB  
```python
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(X_train > 1.0, y_train)
pred = model.predict(X_test > 1.0)

print("Accuracy:", accuracy_score(y_test, pred))
```

8. How do you implement **Support Vector Machines (SVM)**?
  
- Linear SVM  
```python
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X_train, y_train)
```

- Non-linear SVM  
```python
model = SVC(kernel='rbf')
model.fit(X_train, y_train)
```

9. How do you implement a **Random Forest Classifier**?
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

10. How do you implement **Gradient Boosting Methods**?
  
- XGBoost  
```python
from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)
```

- LightGBM  
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier()
model.fit(X_train, y_train)
```

- CatBoost  
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(vervose=0)
model.fit(X_train, y_train)
```
