#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np

class IndependentQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(IndependentQLearningAgent, self).__init__()
        self.State = [(x,y) for x in range(5) for y in range(5)]
        self.State.append(('G',))
        self.State.append(('O',))
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.epsilon = epsilon
        # self.epsilon = 0.1
        # 3 empty lists used to record the episode
        self.logA = 0
        self.logS = 0
        self.R = 0

        self.count = 0

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
        
        if self.curS == ('G',):
            self.count += 1
        print('--------current satte:',self.curS)
        print('-----------', self.count)
        biggest = 0
        for a in self.possibleActions:
            if self.Q[self.curS][a] > biggest:
                biggest = self.Q[self.logS][a]
        # biggest = max(self.Q[self.logS[-1]])
        before = self.Q[self.logS][self.logA]
        self.Q[self.logS][self.logA] += self.learningRate * (self.R + self.discountFactor*biggest - self.Q[self.logS][self.logA])
        
        return  self.Q[self.logS][self.logA] - before


    def act(self):
        comp = {}
        for a in self.possibleActions:
            comp[a] = self.Q[self.curS][a]

        action = [key for key, value in comp.items() if value == max(comp.values())]
        if len(action) == 6:
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

        if state == 'GOAL':
            return ('G',)
        elif state == 'OUT_OF_BOUNDS' or state == 'OUT_OF_TIME':
            return ('O',)
        else:
            return tuple(state[0][0])




    def setState(self, state):
        self.curS = state

        

    def setExperience(self, state, action, reward, status, nextState):
        self.logS = state
        self.logA = action

        self.R = reward
        self.curS = nextState


    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        self.curS = 0
        self.logA = 0
        self.logS = 0
        self.R = 0

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        self.epsilon = (-1/numEpisodes) * episodeNumber + 1
        self.learningRate = self.learningRate#/(episodeNumber+1)
        return self.learningRate, self.epsilon


# class IndependentQLearningAgent(Agent):
# 	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
# 		super(IndependentQLearningAgent, self).__init__()
#
# 	def setExperience(self, state, action, reward, status, nextState):
# 		raise NotImplementedError
#
# 	def learn(self):
# 		raise NotImplementedError
#
# 	def act(self):
# 		raise NotImplementedError
#
# 	def toStateRepresentation(self, state):
# 		raise NotImplementedError
#
# 	def setState(self, state):
# 		raise NotImplementedError
#
# 	def setEpsilon(self, epsilon):
# 		raise NotImplementedError
#
# 	def setLearningRate(self, learningRate):
# 		raise NotImplementedError
#
# 	def computeHyperparameters(self, numTakenActions, episodeNumber):
# 		raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=100)

    args=parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
    agents = []
    for i in range(args.numAgents):
        agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = ["IN_GAME","IN_GAME","IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0
        timeSteps = 0

        while status[0]=="IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies = []
            for agentIdx in range(args.numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(args.numAgents):
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx],
                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation
