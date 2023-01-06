import numpy as np
import BSG
import AI
import randseq
import test
agent = AI.DQNAgent(5, 11)
agent.load("./save/BSG-dqn.h5")
AI = randseq.randseq(10,5)
env = BSG.BigSmallGame(10,5)
AI = randseq.randseq(10,5)
#test.validifying(env, agent, AI)


def validifying(env, agent, randseqAI):
	win = 0
	for e in range(100):
		win1, win2 = 0, 0
		randseqAI.reset()
		[points1, points2, win1, win2, rounds] = env.reset()
		for t in range(5):
			act1, act2 = agent.act(np.reshape([points1, points2, win1, win2, rounds], [1, 5])), randseqAI.act()
			points1, points2, win1, win2, rounds = env.step(act1, act2)
			if t==4:
				if win1>win2:	
					reward1 = 1
					win += 1
				elif win2>win1:	reward2 = 1
				done = 1
			if points1<0:
				reward1 = -1
				done = 1
				continue
			if points2<0:
				reward2 = -1
				done = 1
				win += 1
				continue
	return win
	
