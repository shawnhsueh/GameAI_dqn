# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py
import BSG
import collections
import dqn

EPISODES = 50000
e = 0

env = BSG.BigSmallGame(10, 5)
state_size = 5
action_size = 11
agent = dqn.AI(state_size, action_size)
batch_size = 32
counts = np.zeros((EPISODES / 100, 100))
for e in range(EPISODES):
    state1 = env.reset()
    state2 = state1
    state1 = np.reshape(state1, [1, state_size])
    state2 = np.reshape(state2, [1, state_size])
    done = 0
    for time in range(5):
        points1, points2, win1, win2, rounds = state1[0]  # current state
        state2 = np.array([[points2, points1, win2, win1, rounds]])
        action1, action2 = agent.act(state1), agent.act(state2)
        # print 'action1 = ',action1, 'action2=', action2
        next_state1 = env.step(action1, action2)
        points1, points2, win1, win2, rounds = next_state1  # next state
        reward1, reward2 = 0, 0
        if rounds == 5:
            if win1 > win2:
                reward1 = 1
            elif win2 > win1:
                reward2 = 1
            done = 1
        if points1 < 0:
            reward1 = -1
            done = 1
        if points2 < 0:
            reward2 = -1
            done = 1
        print(next_state1)

        next_state2 = [next_state1[1], next_state1[0], next_state1[3], next_state1[2], next_state1[4]]
        next_state1 = np.reshape(next_state1, [1, state_size])
        next_state2 = np.reshape(next_state2, [1, state_size])

        agent.remember(state1, action1, reward1, next_state1, done)
        agent.remember(state2, action2, reward2, next_state2, done)
        state1 = next_state1
        state2 = next_state2
        if done:
            print("episode: {}/{}, round: {}, tau: {:.2}".format(e, EPISODES, rounds, agent.tau))
            env.reset()
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 1000 == 0:
            agent.save("../checkpoints/BSG-dqn.h5")
