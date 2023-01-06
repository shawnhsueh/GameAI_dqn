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

EPISODES = 10
e = 0

class AI:
    def __init__(self, state_size, action_size):
        self.state_size = state_size #[[my points, their points, my win, their win]]
        self.action_size = action_size #[[WinProbabilityChoosing 1, ..., WinProbabilityChoosing 100]]
        self.memory = deque(maxlen=200)
        self.gamma = 0.95    # discount rate
        self.tau = 10.0  # exploration rate
        self.tau_min = 0.1
        self.tau_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_size, activation='tanh'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
	#([[my points, their points, my win, their win, rounds]], [0~10], -1 or 0 or 1, [[my points, their points, my win, their win]], 1 or 0)

    def act(self, state):
    	if np.random.rand()<0.01*self.tau:
    		remain_points = state[0][0]
    		return np.random.choice(range(remain_points+1)) #random choice of remaining points
        act_values = self.model.predict(state)
	nom = np.exp(act_values/self.tau)
	denom = np.sum(np.exp(act_values/self.tau))
	probability = nom/denom
	#print probability
        return np.random.choice(range(self.action_size), p = probability[0])  # returns action

    def replay(self, batch_size):
	minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            points1 = state[0][0]
            points2 = state[0][1]
            target_f[0][points1+1:] = -1
            if reward==1:
            	if state[0][2]==state[0][3]: # if opponent wins the same rounds as me
            		target_f[0][:points2] = -1
            		target_f[0][points2+1:points1] = 1
            	else:
            		target_f[0][1:] = 1
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.tau > self.tau_min:
            self.tau *= self.tau_decay

    def load(self, name):
        self.model.load_weights(name)
        self.tau = self.tau_min

    def save(self, name):
        self.model.save_weights(name)
    def callpredict(self, state):
	return self.model.predict(state)
    def callp(self, state):
    	act_values = self.model.predict(state)
	nom = np.exp(act_values/self.tau)
	denom = np.sum(np.exp(act_values/self.tau))
	probability = nom/denom
	return probability



env = BSG.BigSmallGame(10, 5)
state_size = 5
action_size = 11
agent = AI(state_size, action_size)
batch_size = 32
counts = np.zeros((EPISODES/100, 100))
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
		action1, action2 = agent.act(state1), agent.act(state2)
		#print 'action1 = ',action1, 'action2=', action2
		next_state1 = env.step(action1, action2)
		points1, points2, win1, win2, rounds = next_state1 #next state
		reward1, reward2 = 0, 0
		if rounds==5:
			if win1>win2:	reward1 = 1
			elif win2>win1:	reward2 = 1
			done = 1
		if points1<0:
			reward1 = -1
			done = 1
		if points2<0:
			reward2 = -1
			done = 1
		print next_state1
		
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
#	counts[e/100][e%100] = rounds
#	if e % 100 == 0:
#		agent.save("./save/BSG-dqn.h5")
#for i in range(EPISODES/100):
#	print collections.Counter(counts[i])
	 

agent = dqn.AI(5, 11)
agent.load("./save/BSG-dqn.h5")
agent.act(np.array([[10,10,0,0,0]]))

print agent.callpredict(np.reshape([2, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([4, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([6, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([3, 2, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([3, 1, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([4, 2, 2, 2, 4], [1, state_size]))
print agent.callpredict(np.reshape([6, 6, 1, 1, 2], [1, state_size]))
print agent.callpredict(np.reshape([5, 3, 2, 1, 3], [1, state_size]))

