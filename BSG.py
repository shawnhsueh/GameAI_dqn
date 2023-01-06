class BigSmallGame:
    def __init__(self, total_points, rounds):
	self.total_points = total_points
	self.rounds = rounds
	self.points1 = self.total_points #100 initial points
	self.points2 = self.total_points
	self.win1 = 0
	self.win2 = 0
	self.even = 0
	self.state = []
    def step(self, num1, num2):
	if num1>num2:
		self.win1 += 1
	elif num2>num1:
		self.win2 += 1
	else:
		self.even += 1
	self.points1 -= num1
	self.points2 -= num2
	self.state = [self.points1, self.points2, self.win1, self.win2, self.win1+self.win2+self.even]
	if self.win1+self.win2+self.even == self.rounds: self.reset()
	return self.state
    def reset(self):
	self.points1 = self.total_points
	self.points2 = self.total_points
	self.win1 = 0
	self.win2 = 0
	self.even = 0
	return [self.total_points, self.total_points, 0, 0, 0]

