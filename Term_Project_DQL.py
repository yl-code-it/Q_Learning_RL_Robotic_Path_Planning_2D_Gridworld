# Reinforcement Learning 3547 - Term Project
# Robot Manipulator Path Planning using Q-Learning and DQN – 2D Grid World Case Study
# Yu Lin
# Presentation date: 12/7/2022
# Algorithm 2 - Deep Q-Learning Network of DRL
# noted that we need RoboDK simulation environment running besides the codes to see the robot actions

import numpy as np
import random
from Gridworld import Gridworld    # Gridworld environment
from robodk import *        # Math toolbox for robots
from robolink import *      # RoboDK's API - the link between RoboDK and Python
import time
import torch
from IPython.display import clear_output
from matplotlib import pylab as plt
from collections import deque

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

# Define robot 2D workspace
# 800 x 800 mm 2d grid, each robot cell is 200 mm
# (1,1): 800, -300, 1100 mm
# (4,4): 800, 300, 500 mm
# The start, target and obstacle positions are random
# for example: obstacle (3,3): 800, 150, 650 mm

# Get the robot poses by name:
start = RDK.Item('Start')
start_pose = start.Pose()
start_x = 800
start_y = -300
start_z = 1100
target = RDK.Item('Target')
target_pose = target.Pose()
target_x = 800
target_y = 300
target_z = 500
obstacle = RDK.Item('Obstacle_Ball')
obstacle_pose = obstacle.Pose()
obstacle_x = 800
obstacle_y = 150
obstacle_z = 650
target_box = RDK.Item('Target_Box')
target_box_pose = target_box.Pose()
target_box_x = 900
target_box_y = 300
target_box_z = 500
move = 200.0
# Y coordinates of each cell - state
Y_vector = [start_y,start_y+move,start_y+move*2,start_y+move*3,
            start_y,start_y+move,start_y+move*2,start_y+move*3,
            start_y,start_y+move,start_y+move*2,start_y+move*3,
            start_y,start_y+move,start_y+move*2,start_y+move*3]
# Z coordinates of each cell - state
Z_vector = [start_z,start_z,start_z,start_z,
            start_z-move,start_z-move,start_z-move,start_z-move,
            start_z-move*2,start_z-move*2,start_z-move*2,start_z-move*2,
            start_z-move*3,start_z-move*3,start_z-move*3,start_z-move*3]

# Move the robot to the start point:
robot.MoveJ(start)

# 4 actions available
action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}

# Define a model of neural network using PyTorch as the Q function
l1 = 48    # state size: 4x4x3=48. tensors of 3 layers, each layer is 4x4
l2 = 150
l3 = 100
l4 = 4      # 4 Q values for each action

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)
loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
gamma = 0.9
epsilon = 0.3

# Deep Q-Learning networks with experience replay
epochs = 5000
losses = []
mem_size = 1000  # A
batch_size = 200  # B
replay = deque(maxlen=mem_size)  # C
max_moves = 50  # D
h = 0
for i in range(epochs):
    game = Gridworld(size=4, mode='random')
    state1_ = game.board.render_np().reshape(1, 48) + np.random.rand(1, 48) / 100.0
    state1 = torch.from_numpy(state1_).float()
    status = 1
    mov = 0
    while (status == 1):
        mov += 1
        qval = model(state1)  # E
        qval_ = qval.data.numpy()
        if (random.random() < epsilon):  # F
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)

        action = action_set[action_]
        game.makeMove(action)
        state2_ = game.board.render_np().reshape(1, 48) + np.random.rand(1, 48) / 100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()
        done = True if reward > 0 else False
        exp = (state1, action_, reward, state2, done)  # G
        replay.append(exp)  # H
        state1 = state2

        if len(replay) > batch_size:  # I
            minibatch = random.sample(replay, batch_size)  # J
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])  # K
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

            Q1 = model(state1_batch)  # L
            with torch.no_grad():
                Q2 = model(state2_batch)  # M

            Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])  # N
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()

        if reward != -1 or mov > max_moves:  # O
            status = 0
            mov = 0
losses = np.array(losses)

# A Set the total size of the experience replay memory
# B Set the minibatch size
# C Create the memory replay as a deque list
# D Maximum number of moves before game is over
# E Compute Q-values from input state in order to select action
# F Select action using epsilon-greedy strategy
# G Create experience of state, reward, action and next state as a tuple
# H Add experience to experience replay list
# I If replay list is at least as long as minibatch size, begin minibatch training
# J Randomly sample a subset of the replay list
# K Separate out the components of each experience into separate minibatch tensors
# L Re-compute Q-values for minibatch of states to get gradients
# M Compute Q-values for minibatch of next states but don't compute gradients
# N Compute the target Q-values we want the DQN to learn
# O If game is over, reset status and mov number


def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


# test the trained DQN model in 2d grid world
def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 48) + np.random.rand(1, 48) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1

    while (status == 1):  # A
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 48) + np.random.rand(1, 48) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win


# move a robot in RoboDK with the trained DQN model
def test_robot(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1, 48) + np.random.rand(1, 48) / 10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1

    # update robot start, obstacle and target positions
    robot_start_index = np.argmax(test_game.board.render_np()[0].reshape(1, 16))
    robot_target_index = np.argmax(test_game.board.render_np()[1].reshape(1, 16))
    robot_obstacle_index = np.argmax(test_game.board.render_np()[2].reshape(1, 16))
    start_pose.setPos([start_x, Y_vector[robot_start_index], Z_vector[robot_start_index]])
    start.setPose(start_pose)
    target_pose.setPos([target_x, Y_vector[robot_target_index], Z_vector[robot_target_index]])
    target.setPose(target_pose)
    target_box_pose.setPos([target_box_x, Y_vector[robot_target_index], Z_vector[robot_target_index]])
    target_box.setPose(target_box_pose)
    obstacle_pose.setPos([obstacle_x, Y_vector[robot_obstacle_index], Z_vector[robot_obstacle_index]])
    obstacle.setPose(obstacle_pose)

    # Move the robot to the start point:
    robot.MoveJ(start)
    action_pose = start_pose

    while (status == 1):  # A
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)  # B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1, 48) + np.random.rand(1, 48) / 10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()

        # move robot accordingly
        xyz_ref = action_pose.Pos()
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

        action_pose.setPos([x, y, z])

        # Move to the action:
        robot.MoveL(action_pose)

        # delay 1 second
        time.sleep(1)

        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break

    win = True if status == 2 else False
    return win


# plot out the loss versus epochs
# plt.figure(figsize=(10,7))
# plt.plot(losses)
# plt.xlabel("Epochs",fontsize=22)
# plt.ylabel("Loss",fontsize=22)

#  Check the % of win cases
max_games = 1000
wins = 0
for i in range(max_games):
    win = test_model(model, mode='random', display=False)
    if win:
        wins += 1
win_perc = float(wins) / float(max_games)
print("Games played: {0}, # of wins: {1}".format(max_games,wins))
print("Win percentage: {}%".format(100.0*win_perc))

# check on a few of random cases and move robot accordingly to verify
for i in range(5):
    test_robot(model, mode='random')
