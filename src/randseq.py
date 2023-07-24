import numpy as np

class AI:
	def __init__(self, points, rounds):
		self.points = points
		self.rounds = rounds
		self.current_round = 0
		self.interval = []
		
	def act(self):
		if self.current_round == 0:
			wall = np.random.choice(range(self.points + 1), size = self.rounds - 1) #from 11 intervals choose 4 places
			wall = np.sort(wall)
			wall = np.insert(wall, 0, 0)
			wall = np.insert(wall, self.rounds, self.points)
			self.interval = np.diff(wall)
		action = self.interval[self.current_round]
		self.current_round += 1
		if self.current_round == 5:
			self.current_round = 0
		return action
		
	def reset(self):
		self.current_round = 0
