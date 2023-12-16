# Multi Modal Path Optimization
Our goal is solve the Vehicle Routing Problem, which can be seen as a generalization of the famous Travelling Salesman Problem, using Reinforcement Learning.
The Vehicle Routing Problem can be best understood via an example.
Take any delivery company. Their aim is to deliver products to as many people as fast as possible. They have multiple trucks in the area, going to multiple locations for the dropoffs. So our ultimate goal is really to find a optimized path for each truck, or vehicle, in order to traverse set locations while minimizing time.

For this, we have decided to use MADDPG (Multi Agent Deep Deterministic Policy Gradient).


## Structure 



## FAQs

### What is MADDPG
### What is DDPG
### Why MADDPG

For this, we have used a multi agent variant of DDPG (Deep Deterministic Policy Gradient) known as MADDPG. DDPG itself is an intermediate RL model between Deep Q-Learning and Proximal Policy Optimization. It follows the same principles of Q-Learning while having an Actor-Critic structure. In this structure, the Actor executes the actions and the Critic trains the Actors. You can think of a football team, where the football players are playing or executing the gameplay, while the Coach is observing the players and critizing them on where they could improve.
