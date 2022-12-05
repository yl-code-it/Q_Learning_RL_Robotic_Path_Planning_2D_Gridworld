# Q_Learning_RL_Robotic_Path_Planning_2D_Gridworld
Robot Manipulator Path Planning using Q-Learning and DQN – 2D Grid World Case Study

Yu Lin

How to apply the Reinforcement Learning (RL) of grid world to the topic of path planning of robotic manipulators? 

Recently, a paper was published about Computer Vision-Based Path Planning for Robot Arms in Three-Dimensional Workspaces Using Q-Learning and Neural Networks [1].

In RL domain, the agent is a robot manipulator end-effector in this paper. The environment is a 3D space that has a robot agent, starting pose, target pose – green cube, and an obstacle – red sphere between them to avoid. 

As a term project of RL course, a 2d case study was attempted to plan optimum actions using both classical Q-Learning (QL) of RL and Deep Q-learning Network (DQN) from the starting pose to the target pose without hitting the obstacle. The study used an industrial robotic simulation tool of RoboDK [2] and a 2D grid world environment [3]. 

The Q-Learning is a model-free value-based Reinforcement Learning algorithm. The Q-Table can be used to find the best action for each state. As long as the target and obstacle remained static, the agent could re-use the Q-Table. However, once the positions changed, the Q-table would need to re-learned.

In comparison, the DQN with experience replay has the ability to play random games with the trained deep neural network as the Q-function. Within 1000 random tests, 991 games won with 99.1% of win rate. In the 2nd video below, 5 random robot path planning tests were presented using the trained DQN model.

The study can be further extended into 3D work space of robotic manipulators. 


4 x 4 2D Grid world with an industrial manipulator environment - start, target, obstacle:

Start – (1, 1); 
Target – (4, 4); 
Obstacle – (3, 3)

![image](https://user-images.githubusercontent.com/82203828/203307909-584e0fc7-d9ec-42e0-9dc3-0909b079951b.png)


1). Classical Q-Learning Results:

https://user-images.githubusercontent.com/82203828/203312792-52d7f004-36b5-463a-acc1-c4418d037570.mp4


2). DQN with Experience Replay Results:

https://user-images.githubusercontent.com/82203828/203637947-973a7ea0-d64a-48c5-b457-51681da0014a.mp4

Loss function chart:

![DQN_Loss_chart_1](https://user-images.githubusercontent.com/82203828/203638749-c6ff7b58-a45b-41cf-a878-db15177535e5.PNG)



There are many 2d grid environments available to explore. Here are a few examples of them: 

(1) Gridworld [3]

(2) OpenAI Gym – Frozen Lake [4]

(3) Wumpus world



References:

[1]. Abdi, A.; Ranjbar, M.H.; Park, J.H. Computer Vision-Based Path Planning for Robot Arms in Three-Dimensional Workspaces Using Q-Learning and Neural Networks. Sensors 2022, 22, 1697. https://doi.org/10.3390/s22051697

[2]. RoboDK tools for simulating and programming industrial robots. https://pypi.org/project/robodk/  https://robodk.com/doc/en/PythonAPI/intro.html

[3]. Alexander Zai and Brandon Brown, Deep Reinforcement Learning In Action https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master/Chapter%203 
source of codes of 2d grid world, and Deep Q-learning Network.

[4]. Thomas Simonini, Q-Learning with Frozenlake – OpenAI Gym. https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake_unslippery%20(Deterministic%20version).ipynb

![image](https://user-images.githubusercontent.com/82203828/203307298-75845bdc-598f-4565-9a5a-4bff114ae72b.png)
