# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py
import collections
import sys
sys.path.append('../src/')
import BSG
import dqn

state_size = 5
action_size = 11

agent = dqn.AI(5, 11)
agent.load("../checkpoints/rand/BSG-dqn-beatrand_30100.h5")

print agent.callpredict(np.reshape([2, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([4, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([6, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([3, 2, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([3, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([4, 2, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([6, 6, 1, 1, 2], [1, state_size]))
print agent.callpredict(np.reshape([5, 3, 2, 1, 3], [1, state_size]))

