# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import h5py
import BSG
import collections
import dqn
import randseq

env = BSG.BigSmallGame(10, 5)
state_size = 5
action_size = 11
agent = dqn.AI(state_size, action_size)
agent_rand = randseq.AI(10, 5)
batch_size = 5
EPISODES = 50000
counts = np.zeros((EPISODES/100, 100))
wincounts = np.zeros((EPISODES/100, 100))
agent.load("./save/BSG-dqn.h5")

for e in range(EPISODES):
	state1 = env.reset()
	state2 = state1
	state1 = np.reshape(state1, [1, state_size])
	state2 = np.reshape(state2, [1, state_size])
	done = 0
	for time in range(5):
		points1, points2, win1, win2, rounds = state1[0] #current state
		state2 = np.array([[points2, points1, win2, win1, rounds]])
		action1, action2 = agent.act(state1), agent_rand.act()
		#action1, action2 = agent.act(state1), agent.act(state2)
		#print 'action1 = ',action1, 'action2=', action2
		next_state1 = env.step(action1, action2)
		points1, points2, win1, win2, rounds = next_state1 #next state
		if rounds==5 or points1<0 or points2<0:
			done = 1
		#print next_state1
		
		next_state2 = [next_state1[1], next_state1[0], next_state1[3], next_state1[2], next_state1[4]]
		next_state1 = np.reshape(next_state1, [1, state_size])
		next_state2 = np.reshape(next_state2, [1, state_size])
		
		agent.remember(state1, action1, next_state1, done)
		#agent.remember(state2, action2, next_state2, done)
		state1 = next_state1
		state2 = next_state2
		if done:
			print("episode: {}/{}, round: {}, tau: {:.2}".format(e, EPISODES, rounds, agent.tau))
			env.reset()
			agent_rand.reset()
			break
		if len(agent.memory) >= batch_size:
			agent.replay()
	counts[e/100][e%100] = rounds
	if points1>=0 and win1 > win2:
		wincounts[e/100][e%100] = 1
	if e % 100 == 0:
		agent.save("./save/BSG-dqn.h5")
		print collections.Counter(wincounts[e/100-1])
for i in range(EPISODES/100):
	print collections.Counter(counts[i])
for i in range(EPISODES/100):
	print collections.Counter(wincounts[i])


print agent.callpredict(np.reshape([2, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([4, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([6, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([10, 10, 0, 0, 0], [1, state_size]))
print agent.callpredict(np.reshape([3, 2, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([3, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([4, 2, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([6, 6, 1, 1, 2], [1, state_size]))
print agent.callpredict(np.reshape([5, 3, 2, 1, 3], [1, state_size]))
