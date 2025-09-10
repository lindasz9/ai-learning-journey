# Hybrid Learning Techniques - Test

1. What are hybrid learning techniques?
> Techniques that combine elements of multiple ML categories including supervised, unsupervised and reinforcement learning in order to create more powerful AI systems.

2. What are the types of hybrid learning techniques?
> There's semi-supervised learning, self-supervised learning, transfer-learning, meta-learning, multi-task learning and multi-modal learning.

3. How do these hybrid learning techniques work? For what do we use them? What do the attached concepts mean?

Semi-Supervised Learning  
> A hybrid learning technique taht falls between supervised and unsupervised learning. It uses a small amount of labeled, and a large amount of unlabeled data. First, we train the model on the labeled data, then we infer labels (pseudo-labeling) on unlabeled data. Then we retrain the model using both the original and pseudo-labeled data. Used for text classification, image classification, speech recognition or fraud detection.

- Self training
> A method where the model trains on its own predictions on unlabeled data.

- Co-training
> Two models learn from different views of the data and teach each other.

- Label propagation
> A method that spreads known labels through a graph to predict labels for unlabeled data.

Self-Supervised Learning  
> A hybrid learning technique that creates training pairs from the input data itself, so the model can learn without manual labels. First, it constructs pretext tasks, then the model is trained on this task using unsupervised data. Finally, the model gets updated, then repeat. We can use it for masked word prediction, next sentence prediction, or colorization, mainly for predicting future segments from past signals.

- Pretext task
> A self-made task where the data itself provides the labels for the model to learn useful features.

Transfer Learning
> A hybrid learning technique that reuses a pretrained model on a new but related task. It tarnsfers knowledge from one domain to another to save training time. So first we import the pretrained model, then we fine-tune it on a smaller, specific dataset. 

Meta-Learning (Learning to Learn)
> A hybrid learning technique that trains models in a way that it can adapt to new tasks quickly. The model is more optimized for adaptibility rather than just performance on one dataset. First, we train the model on different kind of tasks, and during training, the model learns a meta-policy that allows quick adaptation. Then, when presented with a new task, it fine-tunes with only a few examples.

- Meta-policy
> A higher-level strategy that learns to select the appropriate policy for different tasks.

Multi-task Learning (MTL)
> A hybrid learning technique where a single model is trained to perform multiple related tasks simultaneously. It learns from different kinds of data and can perform different types of actions. First, the tasks are defined, then a shared backbone (the same neural network layers for all tasks) is used, with task-specific heads (different output layers) attached to handle each individual task. The model is trained using a combined loss function that accounts for all tasks.

- Hard parameter sharing
> Most layers are shared across tasks.

- Soft parameter sharing
> Each task has its own model and its own neural network, but parameters are regularized to be similar.

Multi-modal Learning
> A hybrid learning technique that builds models that can understand information from multiple data modalities rather then relying on a single type of input. First, we encode each modality using its own specialized encoder, then we fuse the outputs (features) into a shared representation, then we use the combined data to predict.

- Cross-modal attention
> A mechanism allowing the model to also consider information from other modalities.

- Fusion strategies
> Methods for combining information from multiple modalities.

- Early fusion
> Combining data before modeling.

- Late fusion
> Processing each source separately and merging results later.

- Hybrid fusion
> Mixing early and late fusion to leverage both advantages.

4. What do these concepts mean?

Backbone
> The main part of a neural network tahte xtracts general features from input data.

Domain
> The type of data and its distribution that a model is trained.

Few-shot learning
> A learning setup where the model learns from only a few labeled examples.

Modality
> The type of input data.

Pseudo-learning
> Assigning artificial labels to unlabeled data using a trained model.

Zero-shot learning
> The model performs a task without having seen any labeled examples for that task during training.
