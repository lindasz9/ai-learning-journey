# Artificial Neural Networks (ANN)

## 🧠 What are Artificial Neural Networks?

Artificial Neural Networks (ANNs) are the foundational models of deep learning. They are inspired by the structure and function of the human brain, where *neurons* process and transmit information through electrical signals. ANNs aim to replicate this behavior through computational units organized in layers.

ANNs learn patterns from data and can model complex, non-linear relationships. They form the basis of many modern AI applications, including image recognition, natural language processing, and predictive analytics.

---

## 🧱 Structure of a Neural Network

### 🔹 *Neurons* (Nodes)

Each *neuron* receives one or more inputs, applies a *weight* to each, sums them, adds a *bias*, and passes the result through an *activation function* to produce an output.

### 🔹 Layers

- **Input Layer**: Receives raw input features.
- **Hidden Layers**: Perform computations using weights, biases, and activation functions.
- **Output Layer**: Produces the final prediction or classification.

### 🔹 Activation Functions

Introduce non-linearity, allowing the network to model complex relationships. Without activation functions, the model behaves like a linear regression.

### 🔹 Forward Propagation

The process by which input data is passed through the network layer by layer, with outputs of one layer becoming inputs to the next, until a prediction is made.

<img src="https://www.marktechpost.com/wp-content/uploads/2022/09/Screen-Shot-2022-09-23-at-10.46.58-PM-1024x499.png" height="300"/>

---

## 🏋️ Training Neural Networks

### 🔹 Loss Functions

Measure the difference between the predicted output and the actual label. Common examples include:

- Mean Squared Error (MSE)
- Cross-Entropy Loss

### 🔹 Backpropagation

An algorithm used to compute the gradient of the loss function with respect to each weight. Gradients are propagated backward from the output to the input layer.

### 🔹 Gradient Descent

An optimization algorithm that updates weights by moving in the direction of the negative gradient to minimize the loss function.

### 🔹 Training Parameters

- **Epoch**: One full pass through the training dataset.
- **Batch size**: Number of samples processed before updating weights.
- **Learning rate**: Controls how much the weights are adjusted during training.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*SCz0aTETjTYC864Bqjt6Og.png" height="300"/>

---

## 🧮 Types of Neural Networks

### 🔹 Feedforward Neural Networks (FNNs)

The simplest type of ANN. Information flows in one direction—from input to output—without loops.

### 🔹 Deep Neural Networks (DNNs)

Feedforward networks with multiple hidden layers, allowing them to learn more abstract representations.

- **Shallow**: One or two hidden layers.
- **Deep**: Three or more hidden layers.

---

## ⚖️ Strengths and Weaknesses of ANNs

### ✅ Strengths

- Can approximate any function (*Universal Approximation Theorem*)
- Handle both structured and unstructured data
- Automatically extract features from raw data

### ❌ Weaknesses

- Require large amounts of data and compute
- Prone to overfitting
- Poor interpretability (*"black-box" models*)

---

## 🧪 Popular Activation Functions

### 🔹 Sigmoid

Outputs values between 0 and 1. Good for probabilities, but suffers from *vanishing gradients*.

<img src="https://hvidberrrg.github.io/deep_learning/activation_functions/assets/sigmoid_function.png" height="300"/>

### 🔹 ReLU (Rectified Linear Unit)

Replaces negative values with zero. Fast and effective, but can lead to "dead *neurons*."

<img src="https://raw.githubusercontent.com/krutikabapat/krutikabapat.github.io/master/assets/ReLU.png" height="300"/>

### 🔹 Leaky ReLU

A variation of ReLU that allows a small, non-zero gradient for negative inputs, which helps avoid dead *neurons*. It outputs a small negative slope (like 0.01 * x) instead of zero for negative values.

<img src="https://www.researchgate.net/publication/376410479/figure/fig15/AS:11431281211482860@1702404866700/A-plot-of-the-leaky-ReLU-activation-function-with-leak-factor-1-10-and-the-ReLU.png" height="300"/>

### 🔹 Tanh

Outputs values between -1 and 1. Zero-centered, but also prone to *vanishing gradients*.

<img src="https://eeyorelee.github.io/img/tanh.png" height="300"/>

### 🔹 Softmax

Converts raw scores into probabilities that sum to 1. Used in multi-class classification.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20221013120722/1.png" height="300"/>

---

## 🧠 Important Concepts

- **Neuron**: A basic computational unit in a neural network that receives inputs, applies weights and a bias, passes the result through an activation function, and produces an output.
- **Universal Approximation Theorem**: A feedforward neural network with a single hidden layer can approximate any continuous function under certain conditions.
- **Vanishing Gradient**: A problem during training where gradients become too small to update weights effectively.
- **Black-box Model**: A model whose internal workings are not easily interpretable, even though it may make accurate predictions.

---
