# Convolutional Neural Networks (CNN)

## 🧠 What are Convolutional Neural Networks?

Convolutional Neural Networks (CNNs) are a specialized type of neural network designed to process and analyze visual data.  
They are particularly effective at capturing spatial hierarchies in images by learning local patterns (like edges, textures, and shapes) and combining them into more complex structures as the network deepens.

CNNs are widely used in tasks where understanding the **spatial** relationships between pixels is critical — such as in computer vision.

---

## 🔍 Core Building Blocks of CNNs

### 🔹 *Convolution* Layer

- Applies a *filter* (or *kernel*) that slides over the input image to extract important features.  
- The operation results in a *feature map* that highlights the presence of patterns in specific regions.  
- **Key components**:
  - **Kernel/filter**: A small matrix of weights that slides over the input image to detect features like edges, textures, or patterns by computing dot products.
  - **Stride**: Controls how much the filter moves at each step.
  - **Padding**: Adds extra pixels around the input to control spatial dimensions.
  - **Dilation**: Increases the spacing within the *kernel* to capture a larger context.
  - **Patch**: A small region of the input image that the *kernel* compares itself to during *convolution*.

<img src="https://i0.wp.com/www.brilliantcode.net/wp-content/uploads/2019/08/CNN_Tutorial_stride.png?resize=800%2C713&ssl=1" height="300"/>

---

### 🔹 Activation Function

- Introduces **non-linearity** into the network, allowing it to learn complex patterns.
- **Most commonly used**:
  - **ReLU** (Rectified Linear Unit): Converts all negative values to zero.
  - **LeakyReLU**: Allows a small gradient when the unit is not active.

<img src="https://miro.medium.com/max/1400/1*v88ySSMr7JLaIBjwr4chTw.jpeg" height="300"/>

---

### 🔹 Pooling Layer

- Reduces the spatial dimensions of the feature maps.
- **Helps in**:
  - Lowering computational load
  - Controlling overfitting
  - Making the model more invariant to translations
- **Types**:
  - **Max Pooling**: Takes the maximum value from each patch.
  - **Average Pooling**: Computes the average value of the patch.

<img src="https://scientistcafe.com/ids/images/poolinglayer.png" height="300"/>

---

### 🔹 Fully Connected Layer

- After *convolution* and pooling layers, the output is *flattened* and passed through **fully connected (dense)** layers.  
- These layers perform high-level reasoning and classification.

<img src="https://pub.mdpi-res.com/symmetry/symmetry-14-00658/article_deploy/html/images/symmetry-14-00658-g001.png?1648101345" height="300"/>

---

## 🧱 CNN Architecture

CNNs are built by stacking layers in a hierarchical fashion.  
A typical CNN follows this sequence:

**[Input Image] → [Conv Layer + Activation] → [Pooling] → (repeat) → [Fully Connected Layer] → [Output]**

### 🏗️ Popular Architectures:

- **LeNet-5**: Early CNN model used for digit recognition.
- **AlexNet**: Brought deep CNNs to popularity by winning ImageNet in 2012.
- **VGG**: Used multiple 3x3 convolution filters stacked together for better feature extraction.

<img src="https://editor.analyticsvidhya.com/uploads/90650dnn2.jpeg" height="300"/>

---

## 📊 CNN Hyperparameters

Designing a CNN involves tuning several hyperparameters:

- **Filter size**: Controls the receptive area of *convolution* (e.g. 3x3, 5x5).
- **Number of filters**: More filters → richer feature representation.
- **Stride**: Affects *downsampling*.
- **Padding**: Affects spatial size of output.
- **Pooling size** (e.g. 2x2): Controls spatial reduction.
- **Number of layers**: Depth increases learning capability but also complexity.

---

## 🧠 Core Concepts

- **Receptive field**: The region of the input that a particular feature is influenced by.
- **Parameter sharing**: Same filter used across image reduces parameter count.
- **Local connectivity**: Each *neuron* connects only to a small region of the input.
- **Translation invariance**: The ability of CNNs to recognize patterns regardless of their position in the input.
- **Hierarchy of features**: Lower layers learn simple patterns (edges), while deeper layers capture complex structures (objects or shapes).

---

## 📘 Applications of CNNs

CNNs are the backbone of modern computer vision.  
Examples of applications include:

- **Image Classification**: Labeling entire images.
- **Object Detection**: Identifying and locating objects in images.
- **Face Recognition**: Matching or verifying faces.
- **Medical Imaging**: Detecting anomalies in scans like X-rays or MRIs.

---

## 🧠 Important Concepts

- **Convolution**: The process of sliding a *filter* over the input to extract features.
- **Downsampling**: Reducing the spatial dimensions to keep important information and lower computation.
- **Feature map**: Output of a *filter* applied to the input.
- **Flatten**: Converts a multi-dimensional *feature map* into a 1D vector for dense layers.
