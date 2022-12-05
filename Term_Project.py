# Reinforcement Learning 3547 - Term Project
# Robot Manipulator Path Planning using Q-Learning and DQN – 2D Grid World Case Study
# Yu Lin, Soumitra Dinda
# Presentation date: 12/7/2022
# Algorithm 1 - Classical Q-Learning Algorithm of RL
# noted that we need RoboDK simulation environment running besides the codes to see the robot actions

import numpy as np
import random
from Gridworld import Gridworld    # Gridworld environment
from robodk import *        # Math toolbox for robots
from robolink import *      # RoboDK's API - the link between RoboDK and Python
import time

# Create a robot manipulator simulator
# Start the RoboDK API:
RDK = robolink.Robolink()

# items definition
ITEM_TYPE_STATION=1 # station item (.rdk files)
ITEM_TYPE_ROBOT=2 # robot item (.robot files)
ITEM_TYPE_FRAME=3 # reference frame item
ITEM_TYPE_TOOL=4 # tool item (.tool files or tools without geometry)
ITEM_TYPE_OBJECT=5 # object item (.stl, .step, .iges, ...)
ITEM_TYPE_TARGET=6 # target item
ITEM_TYPE_PROGRAM=8 # program item (made using the GUI)
ITEM_TYPE_PROGRAM_PYTHON=10 # Python program or macro

# Get the robot item by name:
robot = RDK.Item('ABB IRB 1600-6/1.45', ITEM_TYPE_ROBOT)

# Get the robot poses by name:
home = RDK.Item('Home')
start = RDK.Item('Start')       # 800, -300, 1100 mm
target = RDK.Item('Target')     # 800, 300, 500 mm; obstacle: 800, 150, 650 mm
target_pose = start.Pose()

# Move the robot to the start point:
robot.MoveJ(start)

# Create a grid world 4 x 4 environment
grid_2d = Gridworld(size=4, mode='static')
print("Gridworld_2D")
print(grid_2d.board.render())

# Create the Q-table and initialize it
# 3D grid workspace - 8x8x8=512 states/cells
# actions: forward(F), backward(B), up(U), down(D), left(L), right(R).
# 2D simplified case study - 4 x 4 grid, 4 actions - up, down, left, right
state_size = 16
action_size = 4

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

# Q-table, size of 16x4, rows - states, columns - actions
qtable = np.zeros((state_size, action_size))

# Define hyper-parameters
total_episodes = 20000      # Total episodes
learning_rate = 0.01        # Learning rate
max_steps = 99              # Max steps per episode
gamma = 0.9                 # Discounting rate
epsilon = 1.0               # Exploration rate
max_epsilon = 1.0           # Exploration probability at start
min_epsilon = 0.01          # Minimum exploration probability
decay_rate = 0.001          # Exponential decay rate for exploration prob

# Q-Learning algorithm
# List of rewards
rewards = []

# For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    grid_2d = Gridworld(size=4, mode='static')
    state = np.argmax(grid_2d.board.render_np()[0].reshape(1,16))
    step = 0
    total_rewards = 0

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action_ = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action_ = np.random.randint(0,4)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        action = action_set[action_]
        grid_2d.makeMove(action)  # K
        new_state = np.argmax(grid_2d.board.render_np()[0].reshape(1,16))
        reward = grid_2d.reward()

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        q_current = qtable[state, action_]
        q_updated = q_current + learning_rate * (
                    reward + gamma * np.max(qtable[new_state, :]) - q_current)

        qtable[state, action_] = q_updated

        total_rewards = total_rewards + reward

        # Our new state is state
        state = new_state

        # If done (if we're dead) : finish episode
        if reward != -1:
            break

    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)
print(epsilon)
print(np.argmax(qtable,axis=1).reshape(4,4))

# use the q-table to move the robot
# Reset the environment
grid_2d = Gridworld(size=4, mode='static')
move = 200.0 # each robot cell is 200 mm

for step in range(max_steps):
    # get the state
    state = np.argmax(grid_2d.board.render_np()[0].reshape(1, 16))

    # get the action based on the q-table
    action_ = np.argmax(qtable[state, :])
    action = action_set[action_]
    grid_2d.makeMove(action)
    reward = grid_2d.reward()

    # move robot accordingly
    xyz_ref = target_pose.Pos()
    if action == "u":
        # Calculate the new position around the reference:
        x = xyz_ref[0]
        y = xyz_ref[1]
        z = xyz_ref[2] + move  # new Z coordinate
    elif action == "d":
        x = xyz_ref[0]
        y = xyz_ref[1]
        z = xyz_ref[2] - move  # new Z coordinate
    elif action == "l":
        x = xyz_ref[0]
        y = xyz_ref[1] - move  # new Y coordinate
        z = xyz_ref[2]
    elif action == "r":
        x = xyz_ref[0]
        y = xyz_ref[1] + move  # new Y coordinate
        z = xyz_ref[2]

    target_pose.setPos([x,y,z])

    # Move to the new target:
    robot.MoveL(target_pose)

    # delay 1 second
    time.sleep(1)

    # If done (if we're dead) : finish episode
    if reward != -1:
        break