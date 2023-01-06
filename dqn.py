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

class AI:
	def __init__(self, state_size, action_size):
		self.state_size = state_size #[[my points, their points, my win, their win]]
		self.action_size = action_size #[[WinProbabilityChoosing 1, ..., WinProbabilityChoosing 100]]
		self.memory = deque(maxlen=5)
		self.gamma = 0.95	# discount rate
		self.tau = 10.0  # exploration rate
		self.tau_min = 0.1
		self.tau_decay = 0.9999
		self.learning_rate = 0.001
		self.model = self._build_model()

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(50, input_dim=self.state_size, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(self.action_size, activation='tanh'))
		model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, next_state, done):
		self.memory.append((state, action, next_state, done))
		#([[my points, their points, my win, their win, rounds]], [0~10], -1 or 0 or 1, [[my points, their points, my win, their win]], 1 or 0)

	def act(self, state):
		if np.random.rand()<0.1*self.tau:
			remain_points = state[0][0]
			return np.random.choice(range(remain_points+1)) #random choice of remaining points
		#act_values = self.model.predict(state)
		#nom = np.exp(act_values/self.tau)
		#denom = np.sum(np.exp(act_values/self.tau))
		#probability = nom/denom
		#return np.random.choice(range(self.action_size), p = probability[0])  # returns action
		return np.argmax(self.model.predict(state))
		
	def replay(self):
		minibatch = np.flip(self.memory, 0)
		
		for state, action, next_state, done in minibatch:
			target = self.model.predict(state) #original set of probability
			#points1, points2, win1, win2, rounds = state[0]
			points1, points2, win1, win2, rounds = next_state[0]
			#target[0][points1+1:] = -1 #always not exceed remaining points
			if not done:
				target[0][action] = self.gamma * np.amax(self.model.predict(next_state)[0])
			else:
				if win1>win2: target[0][action] = 1
				if win1==win2: target[0][action] = 0
				if win1<win2: target[0][action] = -1
			if points1<0: target[0][action] = -1
			#elif points1 > points2:
			#	if win1 == win2: # if opponent wins the same rounds as me
			#		target[0][:points2] = -1
			#		target[0][points2+1:points1] = 1
			#	elif win1 > win2:
			#		target[0][1:] = 1
			#	else:
			#		target[0][:points2] = -1
			#		if points1>points2:
			#			target[0][points2+1:points1] = 0
			
			self.model.fit(state, target, epochs=1, verbose=0)
		if self.tau > self.tau_min:
			self.tau *= self.tau_decay

	def load(self, name):
		self.model.load_weights(name)
		#self.tau = self.tau_min

	def save(self, name):
		self.model.save_weights(name)
	def callpredict(self, state):
		return self.model.predict(state)
	#def callp(self, state):
	#	act_values = self.model.predict(state)
	#	nom = np.exp(act_values/self.tau)
	#	denom = np.sum(np.exp(act_values/self.tau))
	#	probability = nom/denom
	#	return probability

