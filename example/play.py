import sys
sys.path.append('../src/')
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
#agent.load("../checkpoints/rand/BSG-dqn-beatrand_30100.h5")
agent.load("../checkpoints/BSG-dqn.h5")

state1 = env.reset()
done = 0
while not done:
    points1, points2, win1, win2, rounds = state1
    state1 = np.reshape(state1, [1, state_size])

    player_bet = input("AI's chip: "+str(points1)+". Your chip: "+str(points2)+". Enter your bet: ")
    rewards = agent.callpredict(state1)
    print "rewards playing from 0 to 10:", rewards

    nom = np.exp(rewards[0][:points1+1])
    denom = np.sum(nom)
    probability = nom/denom
    bet = np.random.choice(range(points1+1), p = probability)

    next_state1 = env.step(bet, player_bet)
    points1, points2, win1, win2, rounds = next_state1

    print "AI's bet: ", bet, "your bet: ", player_bet
    if bet > player_bet:
        print("you lose")
    elif bet < player_bet:
        print("you win")
    else:
        print("tied")

    if rounds == 5 or points1 < 0 or points2 < 0 or win1>(win2+state_size-rounds) or win2>(win1+state_size-rounds):
        done = 1
    state1 = next_state1
    if done:
        if points1 < 0 and points2 < 0:
            print("tied game")
        elif points2 < 0:
            print("you lose the match, you do not have enough chips")
        elif points1 < 0:
            print("you win the match, the AI does not have enough chips")
        elif win1 > win2:
            print("you lose the match")
        elif win2 > win1:
            print("you win the match")
        else:
            print("tied match")
        env.reset()

