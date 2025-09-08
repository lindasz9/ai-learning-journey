# Reinforcement Learning

## 🧠 What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an *agent* learns how to act in an *environment* by performing *actions* and receiving *rewards*.  
Instead of being told the correct answer (like in supervised learning), the agent learns from trial and error.

The core idea is learning a strategy (called a *policy*) to maximize cumulative *reward* over time. This setting is inspired by behavioral psychology and is widely used in fields like robotics, games, and recommendation systems.

---

## 🎯 Key Components of RL

### 🔹 Agent
- The learner or decision maker.
- Interacts with the *environment* to learn how to act.

### 🔹 Environment
- The external system that responds to the *agent’s actions*.
- Provides observations and *rewards*.

### 🔹 State
- A representation of the current situation the *agent* is in.
- Encodes all necessary information to decide what to do next.

### 🔹 Action
- A choice the *agent* makes that affects the *state* of the *environment*.

### 🔹 Reward
- A scalar feedback signal given after each *action*.
- Tells the *agent* how good or bad its *action* was.

### 🔹 Policy
- A strategy used by the *agent* to determine the next *action*.
- Can be deterministic (fixed) or stochastic (random).

### 🔹 Value Function
- Estimates the expected cumulative *reward* from a *state* if following a *policy*.

### 🔹 Q-Function (*Action-Value* Function)
- Estimates the expected cumulative *reward* from a *state*, taking a specific *action* first, and then following a *policy*.

<img src="https://www.j-labs.pl/wp-content/uploads/2023/02/qlearning_02.jpg" height="300"/>

---

## 🔄 The RL Loop

1. The *agent* observes the *state* of the *environment*.
2. It selects an *action* using its current *policy*.
3. The *environment* transitions to a new *state* and gives a *reward*.
4. The *agent* uses this experience to improve its *policy*.
5. Repeat.

<img src="https://www.researchgate.net/profile/Daniel-Araujo-10/publication/337855423/figure/fig3/AS:834401264992256@1575948172072/Basic-diagram-of-a-RL-scheme.png" height="300"/>

### 🧭 *Exploration* vs *Exploitation*
- **Exploration**: Trying new *actions* to discover their effects.
- **Exploitation**: Using known information to maximize *reward*.
- A balance is needed to learn effectively.

---

## 🏁 Types of RL Problems

### 🔹 Episodic vs Continuing Tasks
- **Episodic**: Learning tasks with a clear beginning and end (e.g., games).
  - DQN
- **Continuing**: Tasks that go on indefinitely (e.g., control systems).

### 🔹 Deterministic vs Stochastic Environments
- **Deterministic**: Next *state* is fully determined by current *state* and *action*.
  - Q-Learning, DQN
- **Stochastic**: Outcomes have randomness, requiring probabilistic modeling.
  - Policy Gradient Methods

### 🔹 Off-Policy vs On-Policy
- **Off-policy**: Learning from actions taken by a different *policy*.
  - Q-Learning, DQN, SARSA
- **On-policy**: Learning from actions taken by the current *policy*.
  - Policy Gradient Methods, *Actor*-Critic Methods

### 🔹 Value-based vs Policy-based vs *Actor*-Critic
- **Value-based**: Learns a *value function* (*state* or *state-action* scores) and acts greedily with respect to it.
  - Q-Learning, DQN, SARSA
- **Policy-based**: Learns the *policy* directly as a probability distribution over *actions*, optimizing it to maximize *reward*.
  - Policy Gradient Methods
- **Actor-Critic**: Combines both value-based and *policy*-based methods by having an *actor* (*policy*) that learns the *policy* and a *critic* (*value function*) that evaluates *actions/states* to guide the *Actor*.
  - *Actor*-Critic Methods

### 🔹 Model-free vs Model-based
- **Model-free**: Learns *policies* or *value functions* without an explicit model of the *environment*.
  - Q-Learning, DQN, SARSA, Policy Gradient Methods, *Actor*-Critic Methods
- **Model-based**: Uses a model of the *environment* to plan and improve *policies*.

---

## 🔓 Off-*Policy* Algorithms

### 🔹 Q-Learning
- **How it works**: Learns the *Q-function* that maps *state-action* pairs to expected *rewards*. Updates Q-values using the *Bellman equation*.
- **Bellman equation**: Expresses the value of a state or state-action as the immediate reward plus *discounted* future rewards.
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
  $$
- **When to use**: When the *action space* is limited, and when you don't have to follow the current *policy* (*off-policy*).
- **Strengths**:  
  - Simple to implement  
  - Proven *convergence* under certain conditions  
- **Weaknesses**:  
  - Struggles in high-dimensional or continuous spaces  
  - Needs *exploration* strategy like *ε-greedy*

### 🔹 Deep Q-Networks (DQN)
- **How it works**: Uses deep neural networks to approximate the *Q-function*.
- **When to use**: Large or continuous *state spaces* (e.g., video games).
- **Strengths**:  
  - Handles high-dimensional input (e.g., images)  
  - Achieves human-level performance in some tasks  
- **Weaknesses**:  
  - Harder to train and tune  
  - More computationally intensive  
- **Extra**:
  - **Experience replay**: Stores past experiences to break correlation in data and improve learning stability.
  - **Target network**: A separate copy of the network used to compute stable target Q-values during training.

<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-2019-04-16-at-5.46.01-PM.png" height="300"/>

---

## 🔒 On-*Policy* Algorithms

### 🔹 SARSA (*State-Action-Reward-State-Action*)
- **How it works**: Similar to Q-learning, but updates the Q-values based on the *action* actually taken by the current *policy*.
- **SARSA update rule**:
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]
  $$
- **When to use**: When you want to learn a *policy* that matches the behavior used during learning.
- **Strengths**:  
  - Safer in *environments* with risks  
  - Reflects current *policy’s* behavior  
- **Weaknesses**:  
  - Can be more conservative  
  - Slower *convergence* compared to Q-learning

### 🔹 Policy Gradient Methods
- **How it works**: Directly optimizes the *policy* by adjusting its parameters to maximize expected *reward*.
- **REINFORCE algorithm**:
  $$
  \nabla J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s) Q^{\pi}(s, a) \right]
  $$
- **When to use**: Continuous *action spaces* or when stochastic *policies* are beneficial.
- **Strengths**:  
  - Can learn stochastic *policies*  
  - Suitable for continuous *actions*  
- **Weaknesses**:  
  - High variance in updates  
  - Slower *convergence* than value-based methods

### 🔹 *Actor-Critic* Methods
- **How it works**: Combines *policy gradient* (*actor*) with a *value function* (*critic*) to reduce variance and improve stability.
- **When to use**: Complex *environments* with continuous *action dspace*.
- **Strengths**:  
  - Balances bias and variance  
  - More stable learning than pure *policy* gradients  
- **Weaknesses**:  
  - More components to tune  
  - Can suffer from instability if *critic* is inaccurate  
- **Extra**:
  - **Actor**: The component that proposes *actions* using the *policy*.
  - **Critic**: The component that evaluates the *actor’s actions* using a *value function*.

---

## 📊 Evaluation Metrics

- **Cumulative *Reward***: Total *reward* collected by the *agent* across an *episode* or time period.
- **Average *Reward***: Mean *reward* per step or per *episode* over time.
- **Return**: Sum of future *discounted* *rewards*:
  $$
  G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots
  $$
- **Success Rate**: Ratio of successful *episodes* to total *episodes*.
- **Sample Efficiency**: How well an algorithm learns from limited experience.  
- **Learning Curve**: A graph showing *agent's* performance over training *episodes*.

---

## 🧠 Important Concepts

- ***Action* space**: set of possible *actions*.
- **Convergence**: The point at which an algorithm’s learning stabilizes, meaning its value estimates or *policy* stop changing significantly and consistently lead to optimal or near-optimal decisions.
- **Discounted**: A way of reducing the importance of future rewards by multiplying them with a factor so that immidiate rewards value more.
- **ε-greedy**: An *exploration* strategy where the *agent* chooses a random *action* with probability ε, and the best-known *action* with probability 1−ε. It balances *exploration* and *exploitation*.
- **Episode**: A single sequence of *states*, *actions*, and *rewards* that ends in a terminal *state*. In *episodic* tasks after an *episode*, the *environment* resets.
- ***State* space**: set of possible *states*.
