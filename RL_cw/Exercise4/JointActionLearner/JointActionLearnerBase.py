#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import itertools
import argparse
import numpy as np
		
class JointQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
		super(JointQLearningAgent, self).__init__()	
		self.State = [(x,y) for x in range(5) for y in range(5)]
		self.State.append(('G',))
		self.State.append(('O',))
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.epsilon = epsilon
		# self.epsilon = 0.1
		# 3 empty lists used to record the episode
		self.A = 0
		self.opA = 0
		self.S = 0
		self.R = 0

		self.count = 0

		self.curS = 0



		# Q table is a dict where keys are the "states" and values are another dict.
		# This inside dict contains "actions" as keys and the values are initilised to 0
		self.Q = {}
		self.C = {}
		self.N = {}

		for s in self.State:
			self.Q[s] = {}
			self.C[s] = {}
			self.N[s] = 0
			for a in self.possibleActions:
				self.Q[s][a] = {}
				for opa in self.possibleActions:
					self.Q[s][a][opa] = 0
					self.C[s][opa] = 0

				


	def setExperience(self, state, action, oppoActions, reward, status, nextState):
		self.N[state] += 1
		self.C[state][oppoActions[0]] += 1
		self.S = state
		self.A = action
		self.opA = oppoActions[0]
		self.R = reward
		self.curS = nextState

		
	def learn(self):
		if self.curS == ('G',):
			self.count += 1
		# print('-----learn---current state:',self.curS)
		print('-----------', self.count)		
		sum_list = []
		for a in self.possibleActions:
			cum = 0
			for opa in self.possibleActions:
				if self.N[self.curS] == 0:
					cum += self.Q[self.curS][a][opa] / 6
					sum_list.append(cum)

				else: 
					cum += self.Q[self.curS][a][opa] * self.C[self.curS][opa] / self.N[self.curS]
					sum_list.append(cum)
		biggest = max(sum_list)
		before = self.Q[self.S][self.A][self.opA]
		self.Q[self.S][self.A][self.opA] = (1 - self.learningRate)*self.Q[self.S][self.A][self.opA] + self.learningRate * (self.R + self.discountFactor*biggest)
		
		return  self.Q[self.S][self.A][self.opA] - before

	def act(self):
		comp = {}
		temp = 0
		for a in self.possibleActions:
			for opa in self.possibleActions:
				if self.Q[self.curS][a][opa] >= temp:
					temp = self.Q[self.curS][a][opa]
					comp[a] = self.Q[self.curS][a][opa]
		# print('---table---',comp)
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

	def setEpsilon(self, epsilon) :
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate) :
		self.learningRate = learningRate

	def setState(self, state):
		self.curS = state

	def toStateRepresentation(self, rawState):
		if rawState == 'GOAL':
			return ('G',)
		elif rawState == 'OUT_OF_BOUNDS' or rawState == 'OUT_OF_TIME':
			return ('O',)
		else:
			return tuple(rawState[0][0])
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.epsilon = max(0, (-1/5000) * episodeNumber + 1)
		self.learningRate = self.learningRate#/(episodeNumber+1)
		return self.learningRate, self.epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	numAgents = args.numAgents
	numEpisodes = args.numEpisodes
	for i in range(numAgents):
		agent = JointQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0, numTeammates=args.numAgents-1)
		agents.append(agent)

	numEpisodes = numEpisodes
	numTakenActions = 0

	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
			
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
				agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())

			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			for agentIdx in range(args.numAgents):
				oppoActions = actions.copy()
				del oppoActions[agentIdx]
				agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], oppoActions, 
					reward[agentIdx], status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
