#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import random
import numpy as np

class QLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(QLearningAgent, self).__init__()
        self.State = [(x,y) for x in range(5) for y in range(6)]
        self.State.append("G")
        self.State.append("O")
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.epsilon = epsilon
        # self.epsilon = 0.1
        # 3 empty lists used to record the episode
        self.logA = 0
        self.logS = 0
        self.R = 0


        self.curS = 0
        # self.curA = 0

        # Q table is a dict where keys are the "states" and values are another dict.
        # This inside dict contains "actions" as keys and the values are initilised to 0
        self.Q = {}

        for s in self.State:
            self.Q[s] = {}
            for a in self.possibleActions:
                self.Q[s][a] = 0



    def learn(self):
        biggest = 0
        for a in self.possibleActions:
            if self.Q[self.curS][a] > biggest:

                biggest = self.Q[self.logS][a]

        before = self.Q[self.logS][self.logA]
        self.Q[self.logS][self.logA] += self.learningRate * (self.R + self.discountFactor*biggest - self.Q[self.logS][self.logA])
        return  self.Q[self.logS][self.logA] - before


    def act(self):
        comp = {}
        for a in self.possibleActions:
            comp[a] = self.Q[self.curS][a]

        action = [key for key, value in comp.items() if value == max(comp.values())]
        if len(action) == 5:
            return random.choice(action)
        flag = np.random.binomial(1, 1-self.epsilon)
        not_star = [x for x in self.possibleActions if x not in action]
        # choose optimal action from list 'action'
        if flag == 1:
            result_action = random.choice(action)
        # randomly choose from non-optimal actions, list 'not_star'
        else:
            result_action = random.choice(not_star)
            
        return result_action


    def toStateRepresentation(self, state):
        return state[0]

    def setState(self, state):
        self.curS = state

    def setExperience(self, state, action, reward, status, nextState):
        self.logS = state
        self.logA = action
        # self.logR.append(reward)
        # self.lognexS.append(nextState)
        self.R = reward
        self.curS = nextState
        # self.curA = self.act()

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        self.curS = 0
        # self.curA = 0
        self.logA = 0
        self.logS = 0
        self.R = 0

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        self.epsilon = (-1/numEpisodes) * episodeNumber + 1
        self.learningRate = self.learningRate/(episodeNumber+1)
        return self.learningRate, self.epsilon

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args=parser.parse_args()

    # Initialize connection with the HFO server
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
    numEpisodes = args.numEpisodes

    # Run training using Q-Learning
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = 0
        observation = hfoEnv.reset()

        while status==0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
            update = agent.learn()

            observation = nextObservation
