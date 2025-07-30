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
# Artificial Neural Networks in Python

## Libraries

🧪 **TensorFlow / Keras**
- Classification
- Regression

🧪 **PyTorch**
- Classification
- Regression

---

## 🧪 TensorFlow / Keras

### 🔹 Classification

#### ⚙️ Load and Preprocess Data
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

X, y = load_iris(return_X_y=True)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### 🏗️ Build the Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
```

#### 🧪 Compile and Train
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
```

#### 📊 Evaluate
```python
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", accuracy)
```

---

### 🔹 Regression

#### ⚙️ Load and Preprocess Data
```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### 🏗️ Build the Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

#### 🧪 Compile and Train
```python
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
```

#### 📊 Evaluate
```python
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

## 🧪 PyTorch

### 🔹 Classification

#### ⚙️ Load and Preprocess Data
```python
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
```

#### 🏗️ Build the Model
```python
import torch.nn as nn
import torch.optim as optim

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

#### 🧪 Train the Model
```python
for epoch in range(50):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
```

#### 📊 Evaluate
```python
from sklearn.metrics import accuracy_score

with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test, predicted)
    print("Accuracy:", acc)
```

---

### 🔹 Regression

#### ⚙️ Load and Preprocess Data
```python
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
```

#### 🏗️ Build the Model
```python
import torch.nn as nn
import torch.optim as optim

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Regressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

#### 🧪 Train the Model
```python
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

#### 📊 Evaluate
```python
from sklearn.metrics import mean_squared_error

with torch.no_grad():
    y_pred = model(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
```

```

## **Important** Rules for Notes

- Future notes don’t have to follow this exact format, since topics will naturally require different subtitles and sections. However, this serves as the base structure, which you can adapt as needed for each topic.
- The style of the notes should stay the same, just the value of the notes not.
- You don’t need to keep the exact same subtitles — feel free to modify, add, or remove them depending on what fits the content best.
- **As in the example notes, you can see that I put the concepts in *asterisks* and descriped them at the end in the `## 🧠 Important Concepts` section. You don't need to deal with this, I will do this after you sent me the notes. So just include the `## 🧠 Important Concepts` title at the ned, and leave it empty.**
- **Do not put anything in *single asterisks* or **double asterisks**.**
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
  - ANN in Python
  - Convolutional Neural Networks (CNN)

Next:
- CNN in Python

## File Structure for Deep Learning

Deep Learning Overview
Artificial Neural Networks (ANN)
ANN in Python
Convolutional Neural Networks (CNN)
CNN in Python
Recurrent Neural Networks (RNN)
RNN in Python
Long Short-Term Memory (LSTM)
LSTM in Python
Attention Mechanisms
Transformers
Generative Adversarial Networks (GANs)
Autoencoders
Deep Learning Workflow
Deep Learning in Python