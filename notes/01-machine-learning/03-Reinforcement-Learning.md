# Reinforcement Learning

## 🧠 What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an *agent* learns how to act in an *environment* by performing **action*s* and receiving *rewards*.  
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
- Estimates how good it is to be in a *state* (or to perform an *action*) in terms of expected future *rewards*.

### 🔹 Q-Function (*action-Value Function*)
- Estimates the expected *reward* of taking a specific *action* in a specific *state* and following a *policy* thereafter.

---

## 🔄 The RL Loop

1. The *agent* observes the *state* from the *environment*.
2. It selects an *action* using its current *policy*.
3. The *environment* transitions to a new *state* and gives a *reward*.
4. The *agent* uses this experience to improve its *policy*.
5. Repeat.

### 🧭 Exploration vs Exploitation
- **Exploration**: Trying new *actions* to discover their effects.
- **Exploitation**: Using known information to maximize *reward*.
- A balance is needed to learn effectively.

---

## 🏁 Types of RL Problems

### 🔹 Episodic vs Continuing Tasks
- **Episodic**: Learning tasks with a clear beginning and end (e.g., games).
- **Continuing**: Tasks that go on indefinitely (e.g., control systems).

### 🔹 Deterministic vs Stochastic Environments
- **Deterministic**: Next *state* is fully determined by current *state* and *action*.
- **Stochastic**: Outcomes have randomness, requiring probabilistic modeling.

---

## 🧪 Algorithms Overview

### 🔹 Q-Learning
- **How it works**: Learns the *Q-function* that maps *state-action* pairs to expected *rewards*. Updates Q-values using the Bellman equation.
- **When to use**: Discrete *action* spaces and when you don’t need to follow the current *policy* (*off-policy*).
- **Strengths**:  
  - Simple to implement  
  - Proven convergence under certain conditions  
- **Weaknesses**:  
  - Struggles in high-dimensional or continuous spaces  
  - Needs exploration strategy like ε-greedy
- **Extra**:
  - **Bellman equation**: A recursive equation that describes the value of a *policy* in terms of expected *rewards*.  

---

### 🔹 SARSA (*State-action-Reward-State-action*)
- **How it works**: Similar to Q-learning, but updates the Q-values based on the *action* actually taken by the current policy (on-policy).
- **When to use**: When you want to learn a policy that matches the behavior used during learning.
- **Strengths**:  
  - Safer in environments with risks  
  - Reflects current *policy’s* behavior  
- **Weaknesses**:  
  - Can be more conservative  
  - Slower convergence compared to Q-learning in some cases

---

### 🔹 Deep Q-Networks (DQN)
- **How it works**: Uses deep neural networks to approximate the Q-function. Adds techniques like experience replay and target networks to stabilize training.
- **When to use**: Large or continuous state spaces (e.g., video games).
- **Strengths**:  
  - Handles high-dimensional input (e.g., images)  
  - Achieves human-level performance in some tasks  
- **Weaknesses**:  
  - Harder to train and tune  
  - More computationally intensive

---

### 🔹 Policy Gradient Methods
- **How it works**: Directly optimizes the policy by adjusting its parameters to maximize expected reward.
- **When to use**: Continuous *action* spaces or when stochastic policies are beneficial.
- **Strengths**:  
  - Can learn stochastic policies  
  - Suitable for continuous *actions*  
- **Weaknesses**:  
  - High variance in updates  
  - Slower convergence than value-based methods

---

### 🔹 Actor-Critic Methods
- **How it works**: Combines policy gradient (actor) with a value function (critic) to reduce variance and improve stability.
- **When to use**: Complex environments with continuous *actions*.
- **Strengths**:  
  - Balances bias and variance  
  - More stable learning than pure policy gradients  
- **Weaknesses**:  
  - More components to tune  
  - Can suffer from instability if critic is inaccurate

---

## 📊 Evaluation and Challenges

### 🔹 Reward Shaping
- Designing effective *reward* functions is difficult but crucial.
- Poor design can lead to unintended behaviors.

### 🔹 Credit Assignment Problem
- Figuring out which *action* caused which *reward* when *rewards* are delayed.

### 🔹 Sample Inefficiency
- Many algorithms require millions of steps to learn effective policies.

### 🔹 Stability
- Especially in deep RL, training can be unstable or divergent due to complex interactions between components.

---

## 🧠 Important Concepts

- **Actor**: The component in *actor*-critic algorithms that learns the *policy*.  
- **Critic**: The component in *actor*-critic algorithms that evaluates *actions* using a *value function*.  
- **Deterministic**: A system where the outcome is fully determined by the current state and *action*.  
- **Environment**: The system the agent interacts with in RL, providing states and *rewards*.  
- **Exploration**: Trying new *actions* to discover better strategies.  
- **Exploitation**: Using known strategies that yield high *rewards*.  
- **Episodic task**: An RL task that ends after a finite number of steps.  
- **On-policy**: Learning using the same *policy* that is being improved.  
- **Off-policy**: Learning using a different *policy* from the one being improved.  
- **Policy**: The *agent’s* strategy for choosing *actions*.  
- **Q-function**: Function estimating the value of *state-action* pairs.  
- **Reward**: Scalar feedback given to the agent after taking an *action*.  
- **Sample inefficiency**: When learning requires many inter*actions* to improve.  
- **SARSA**: An on-*policy* RL algorithm that updates Q-values based on the actual *action* taken.  
- **State**: The agent’s perception of the *environment* at a given time.  
- **Stochastic**: Systems with randomness in transitions or *rewards*.  
- **Value function**: Estimates the expected return of being in a *state* or taking an *action*.  
