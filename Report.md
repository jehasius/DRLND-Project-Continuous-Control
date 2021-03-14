# Continuous Control Project - Udacity Deep Reinforcement Learning Expert Nanodegree
#### J. Hampe 03/2021.
---

[1. Introduction](#intro)  
[2. Getting Started](#start)  
[3. Learning Algorithm](#algo)  
[4. Plot of Rewards](#plot)  
[5. Simulation](#sim)  
[6. Ideas for future work](#future)  

[//]: # (Image References)
[image1]: ./pictures/score_episode.png "Score over Episode"
[image2]: ./pictures/aftertraining.png "Trained Agent"
[image3]: ./pictures/roboticarm.gif
[image4]: ./pictures/beforetraining.png "Trained Agent"

<a name="intro"></a>
## 1. Introduction
This project was a part of the Udacity Deep Reinforcement Learning Expert Nanodegree course. The main task of this project, called Continuous Control, was to use PPO, DDPG or other policy based algorithm to train a machine learning agent to control a robotic arm, that has to follow a track in an Unity environment. To find out more about the used Unity environment please look at [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) explainations. The used DDPG - Deep Deterministic Policy Gradient algorithm is an algorithm that was introduced as an "Actor-Critic" method, see paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

### 1.1 The Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

#### Real-World Robotics 
Watch this [YouTube video](https://www.youtube.com/watch?v=ZVIxt2rt1_4) to see how some researchers were able to train a similar task on a real robot! The accompanying research paper can be found [here](https://arxiv.org/pdf/1803.07067.pdf).
![robotic arm][image3]
Training robotic arm to reach target locations in the real world. [(Source)](https://www.youtube.com/watch?v=ZVIxt2rt1_4)

### 1.2 The Observation Space
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
Some information about the environment:
```python
Number of agents: 1
Size of each action: 4
There are 1 agents. Each observes a state with length: 33
The state for the first agent looks like: [ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726671e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
```
### 1.3 AI - Machine Learning - Reinforment Learning
Machine learning, is a branch of artificial intelligence, focuses on learning patterns from data. The three main classes of machine learning algorithms include: unsupervised learning, supervised learning and reinforcement learning. Each class of algorithm learns from a different type of data. You can find a good overview over the main concepts of machine learning at [Unity Background-Machine-Learning](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Background-Machine-Learning.md)

<a name="start"></a>
## 2. Getting Started
To set up and run the environment please follow the generell instructions in the [README.md](./README.md) and jupyter notebook [Continuous_Control.ipynb](./Continuous_Control.ipynb). The Jupyter notebook also contains the whole project algorithms.

<a name="algo"></a>
## 3. Learning Algoritm
The used DDPG - Deep Deterministic Policy Gradient algorithm is an algorithm that was introduced as an "Actor-Critic" method that can be used in the context of continuous action spaces. DDPG is a different kind of actor-critic method, it could be seen as approximate DQN, instead of an actual actor critic. The reason for this is that the critic in DDPG, is used to approximate the maximizer over the Q values of the next state, and not as a learned baseline. The algorithm uses two networks, one for the policy to map states to actions and one for a critic to map state and action pairs as Q-values.

### 3.1. Network Architecture

The two networks, one for the policy to map states to actions and one for a critic to map state and action pairs as Q-values can be found in the [model.py](./model.py) file.

Build an actor (policy) network that maps states -> actions
```python
Actor(
  (fc1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=128, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
)
```
Build a critic (value) network that maps (state, action) pairs -> Q-values
```python
Critic(
  (fcs1): Linear(in_features=33, out_features=128, bias=True)
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=132, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

### 3.2 DRL Hyperparameter
To tune the a Deep Reinforcement Learning system you are always have a bunch of hyperprameter in your algoritm. With this hyperparameters you can influence and optimize the hole learnig prozess. This is a very challenging project and it took me a long time to find the right hyperparameter combination. Thnk to the knowlage base of Udacity I found useful hints and commends to tune the hyperparameters and also the network feature sizes.

```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```
### 3.3 Build and train the DDPG algorithm
The DDPG algorithm and trainig process was realized with an deep learning agent [ddpg_agent.py](./ddpg_agent.py), the model [model.py](./model.py) file and the [Continuous_Control.ipynb](./Continuous_Control.ipynb). The model parameter were saved can be accesed for further tests an explorations.


<a name="plot"></a>
## 4. Plot of Rewards
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.
The Environment was solved in Environment solved in 45 episodes with an average score of 30.43.
```python
Episode 50 	Score: 3.18 	Average Score: 2.51
Episode 100 	Score: 22.29 	Average Score: 11.01
Environment solved in 45 episodes! Average score of 30.43
```
In the picture below you can see the approproate plot of the rewards during the training process as score over episodes.

![alt text][image1]  

## 5. Simulation<a name="sim"></a>
The model was successfully trained and the agent was able to reach target locations.
Picture of the Unity simulation - typical position of the robotic arm before the training. 
![alt text][image4]  

Picture of the Unity simulation - typical position of the robotic arm after the training. 
![alt text][image2]  

<a name="future"></a>
## 6. Ideas for future work
The alorithm is running quit well but we can try to tune hyper-parameters.
Testing different network architecture that uses other network architectures.
Solving the second option with 20 robotic arms with useful for algorithms like: 
[PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience. 

