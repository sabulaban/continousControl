
# Udacity Deep Reinforcement Learning Nano-degree
## Project: P2 - Continous Control


<u>**This report aims at providing low level details of P2-Continous Control implementation. For high level details, and dependencies, please refer to the README.md**</u>

## Dev Environment Introduction
--------
### Chosen Algorithm 

The algorithm of choice for this project is DDPG, reason behind that are:
1- Ability to reuse and expand as much as possible the classes created to use for MADDGP; easier migration to DDPG
2- The action space being a continous space

### Algorithm Details

The algorithm consists of the following classes, each class will be explained in details:

    1- Agent
    2- Memory
    3- Noise
    4- Models

The agent has the following attributes:
    
    1.1 Actor - Local: Running actor network; defined in models
    1.2 Actor - Target: Target; not learning, actor network, identical architecture to Actor - Local
    1.3 Actor - Local Optimizer: Adam optimizer with 1e-3 learning rate
    1.4 Critic - Local: Running critic network; defined in models
    1.5 Critic - Target: Target; not learning, critic network, identical architecture to Critic - Local
    1.6 Critic - Local Optimizer: Adam optimizer with 1e-3 learning rate, and L2 weight decay of 0
    1.7 Noise - Ornstein–Uhlenbeck process based noise
    
And it has the following methods:
    
    1.1 Step: This method takes an exepreince defined as: state, action, reward, next_state, done
        1.1.1 The experience is addedd to the memory buffer
        1.1.2 If memory > batch size (128) and it is a learning step, a batch of 128 experiences is sampled, and then passed to the learn method for updating the actor/critic for 10 times. Steps to learn chosen is 20, and actor/critic are trained for 10 times. 
    1.2 Learn: This method takes an experience, and discount rate gamma, and does the following:
        1.2.1 Update the critic by getting the MSE of Q_local(state_t, action_t) and rewards + gamma * Q_target(state_t+1,action_t+1); where action_t+1 is derived from the target actor for next state.
        1.2.2 Gradient clipping is introduced to stabilize learning, since the agent will be running multiple learning runs every n step.
        1.2.3 Calculate the actor loss as: -Q(states, actor(states).mean()
        1.2.4 Update the actor using the optimizer
        1.2.5 Soft update the actor, where the Actor Target moves by a factor tau (1e-3) towards the Actor Local
        1.2.6 Soft update the critic, where the Critic Target moves by a factor tau (1e-3) towards the Critic Local
        1.2.7 Reset the noise process
        1.2.8 Reduce EPSILON (the noise discount factor) by value EPSILON_DECAY (1e-6)
    1.3 Act: This method takes the state(s) as an input as well as boolean flag for adding noise, and returns action values of size: action_size, and continous values clipped between -1. and 1. If noise flag is enabled, noise is multiplied by a factor EPSILON, which decays over learning steps. 
    1.4 Reset: Resets the noise process
    1.5 Soft Update: Updates the target network by a factor tau 
    
### Algorithm Implementation

- For max episodes n do:
    - Reset environment and get initial state
    - For max steps m or episode done do:
        - Take an action with noise using the local actor for the 20 arms
        - Observe the next states, rewards and dones
        - Take a step using the step method for the agent
        - Set state = next_state
        - If any of the agents is done, break the episode

### Actor Neural Network
The Actor network is defined in models.py. The actor consists of 3 fully connected layers, and batch normalization layer.
The batch normalization is applied to the output of the input layer. All fully connected layers use RELU for activation, except for the output layer which uses TANH for activation.

### Critic Neural Network
The critic network is defined in models.py. The critic consists of 3 fully connected layers, and batch normalization layer. Out of the 3 fully connected layers, there are 2 input layers, input layer 1 is for the state, input layer 2 is for the output of batch normalization of the input layer 1 and the actions.
All fully connected layers use RELU for activation, except for the output layer doesn't use an activation.

### Hyper-parameters

Following is the list of the hyper-parameters used

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 128        # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR_ACTOR = 1e-3         # learning rate of the actor
- LR_CRITIC = 1e-3        # learning rate of the critic
- WEIGHT_DECAY = 0        # L2 weight decay
- LEARN_EVERY = 20        # learning timestep interval
- LEARN_NUM = 10          # number of learning passes
- OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter
- OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter
- EPSILON = 1.0           # explore->exploit noise process added to act step
- EPSILON_DECAY = 1e-6    # decay rate for noise process

Out of those hyper-parameters, it was noticed the following:

- Increasing batch-size will slow down the learning, and no significant improvement, tried buffer sizes: 64, 128, and 1024, and over 10 episodes, 128 performed the best in terms of score / time
- Learn every and learn num turned out to be the most crucial for my experiments. Increasing learning number, increased the episode time significantly; beyond 5 minutes per episode for 20 learning steps every 10 steps. Same applies to deacresing the learn_every factor, so for example, despite the combination learn every 10/learn num 20, outperformed on the short term, but in all cases the agent had to run minimum 100 episodes to solve the challenge, thus, a combination of 20/10 (learn_num/learn_every) lengthens the training time significantly.
- Another combination that was tried was (1/1) and on the short term; 10 episodes, didn't perform as good as (10/20) combination
- It was noticed that the learning is quite resillient to neural network architecture, and learning rate, tried random learning rate between 1e-3 and 1e-5, didn't see an improvement or reduction in performance over the short term (10 episodes)

## Results

<img src="./ddpg.training.png" width="600"/>
- The agent achieved the target in 100 episodes as displayed above

<img src="./reacher.perf" width="400"/>
- The agent stop improving after around the 40th episode

## Discussion and Future Work
-------
- Introduction of Prioterized Experience Replay to improve the learning
- Introduction of early stop of learning, to accelerate the learning, where if learning was stopped after 40th episode + evaluation buffer lenght. 
- The use of different exploration levels for different agents, this requires the use of multiple act functions, one act per agent, or the use of MADDPG

