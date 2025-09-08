# Reinforcement Learning - Test

1. What is reinforcement learning?
> Reinforcement learning is a category of ML that differs from the other two. In this approach, an agent is placed in an environment and takes actions to reach different states. After each action, it receives a reward — a feedback signal indicating how good the action was — and uses these rewards to learn how to act effectively in the environment.

2. What do these key components mean?

Agent
> The decision maker that interacts with the environment to learn how to act effectively.

Environment
> The external system that responds to the agent's actions, and gives rewards.

State
> The current situation where the agent is in, fully describing the environment at a given time.

Action
> A choice that the agent makes which affects the state of the environment.

Reward
> A feedback signal given after each action, that tells the agent how good or bad its action was.

Policy
> The strategy used by the agent to determine the next action.

Value Function
> Estimates the expected cumulative reward from a state if it follows a given policy.

Q-Function (Action-Value Function)
> Estimates the expected cumulative reward from a state, taking a specific action first, and then following a policy.

3. What is the loop in RL?
> First, the agent observes the state of the environment, then it selects an action using its current policy. The environment transitions into a new state and gives a reward. The agent improves its policy based on the reward.

4. What's the difference between exploration and exploitation?
> Exploration means trying new actions to discover their effects, while exploitation means using existing knowledge to maximize rewards. To act effectively in an environment, it’s important to maintain a balance between the two.

5. What do these types of RL problems mean?

Episodic vs Continuing
> Episodic: the learning process has a clear beginning and ending. Continuing: the learning process goes on indefinitely.

Deterministic vs Stochastic
> Deterministic: the next state is fully determined by the current state and action. Stochastic: the next state is also based on randomness.

Off-policy vs On-policy
> Off-policy: learning from actions taken by a different policy. On-policy: learning from actions taken by the current policy.

Value-based vs Policy-based vs Actor-Critic
> Value-based: learns a value function and acts greedily with respect to it. Policy-based: learns the policy directly as a probability distribution over actions. Actor-Critic: combines both value and policy-based methods. The Actor learns the policy, and the Critic evaluates actions or states to guide the Actor.

Model-free vs Model-based
> Model-free: learns policies or value-functions without an explicit model. Model-based: uses a model to plan and improve policies.

6. How do these algorithms work? When do we use them? What are some strengths and weaknesses? What do the attached concepts mean?

Q-Learning
> An off-policy RL algorithm that learns the Q-function, then updates Q-values using the Bellman equation (combining immediate and future rewards). It's used when the set of possible actions is countable and we need an off-policy algorithm. It's simple to implement and it will converge under certain conditions. But it struggles in high-dimensional data or continuous spaces, and it needs an exploration strategy.

- Bellman equation
> Expresses the value of a state or state-action as the immediate reward plus discounted future rewards.

Deep Q-Networks (DQN)
> An off-policy RL algorithm that uses neural networks to approximate the Q-function. It's used in larger continuous state spaces. It can handle high-dimensional data but it's also harder to train and computationally intensive.

- Experience replay
> Stores past experience to break correlation and improve learning stability.

- Target network
> A separate copy of the network used to compute stable target Q-values during training.

SARSA (State-Action-Reward-State-Action)
> An on-policy RL algorithm that is similar to Q-learning, but it updates the Q-values based on the action actually taken by the current policy. Used when an on-policy method is needed. It's safer but converges more slowly.

Policy Gradient Methods
> An on-policy RL algorithm that optimizes the policy directly by adjusting its parameters to maximize expected reward. It's used in continuous action spaces or when policies should be stochastic. It works on continuous actions but converges slowly and high variance is common in updates.

Actor-Critic Methods
> An on-policy RL algorithm that combines policy gradient (actor) with a value function (critic) to improve stability. We use it for complex problems with continuous action space. It's stable, but more complicated.

- Actor
> The component that proposes actions using the policy.

- Critic
> The component that evaluates the actor's action using a value function.

7. What do these evaluation metrics mean?

Cumulative Reward
> Total reward collected by the agent across an episode or time period.

Average Reward
> Mean reward per step or per episode over time.

Return
> Sum of future discounted rewards.

Success rate
> Ratio of successful episodes to total episodes.

Sample Efficiency
> The amount of experience needed to reach a performance threshold.

Learning Curve
> A graph showing the agent's performance over the episodes.

8. What do these concepts mean?

Action space
> Set of possible actions.

Convergence
> The point at which the algorithm's learning stabilizes meaning its policy stops changing and consistently leads to optimal or near-optimal decisions.

Discounted
> A way of reducing the importance of future values by multiplying them with a factor so that immediate rewards value more.

ε-greedy
> An exploration strategy where the agent chooses a random action with probability ε, and the best-known *action* with probability 1−ε.

Episode
> A sequence of states, actions and rewards that ends in a terminal state. After an episode, the environment resets.

State space
> Set of possible states.
