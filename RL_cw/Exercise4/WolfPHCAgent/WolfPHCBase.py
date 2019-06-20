#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.State = [(x,y) for x in range(5) for y in range(5)]
		self.State.append(('G',))
		self.State.append(('O',))
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.winDelta = winDelta
		self.loseDelta = loseDelta

		# self.epsilon = 0.1
		# 3 empty lists used to record the episode
		self.logA = 0
		self.logS = 0
		self.R = 0
		self.curS = 0
		self.curA = 0

		self.count = 0
		# Q table is a dict where keys are the "states" and values are another dict.
		# This inside dict contains "actions" as keys and the values are initilised to 0
		self.C = {}
		self.Q = {}
		self.policy = {}
		self.ave_po = {}
		for s in self.State:
			self.Q[s] = {}
			self.policy[s] = {}
			self.ave_po[s] = {}
			self.C[s] = 0
			for a in self.possibleActions:
				self.Q[s][a] = 0
				self.policy[s][a] = 1/6
				self.ave_po[s][a] = 0
	
		
	def setExperience(self, state, action, reward, status, nextState):
		self.logA = action
		self.logS = state
		self.R = reward
		self.curS = nextState

	def learn(self):
		if self.curS == ('G',):
			self.count += 1
		print('--------current satte:',self.curS)
		print('-----------', self.count)		
		biggest = 0
		for a in self.possibleActions:
			if self.Q[self.curS][a] > biggest:
				biggest = self.Q[self.curS][a]
		before = self.Q[self.logS][self.logA]
		self.Q[self.logS][self.logA] += self.learningRate * (self.R + self.discountFactor * biggest - self.Q[self.logS][self.logA])
		return self.Q[self.logS][self.logA] - before
			 

	def act(self):
		comp = {}
		for a in self.possibleActions:
			comp[a] = self.Q[self.curS][a]

		action = [key for key, value in comp.items() if value == max(comp.values())]

		not_star = [x for x in self.possibleActions if x not in action]

			
		return random.choice(action)

	def calculateAveragePolicyUpdate(self):
		ave_list = []
		for a in self.possibleActions:
			self.ave_po[self.logS][a] += (1/self.C[self.logS]) * (self.policy[self.logS][a] - self.ave_po[self.logS][a])
			ave_list.append(self.ave_po[self.logS][a])
		return ave_list

	def calculatePolicyUpdate(self):
		delta = 0
		policy_update_list = {}
		result = []
		for a in self.possibleActions:
			policy_update_list[a] = 0
		pai = 0
		pai_bar = 0
		for a in self.possibleActions:
			pai += self.policy[self.logS][a] * self.Q[self.logS][a]
			pai_bar += self.ave_po[self.logS][a] * self.Q[self.logS][a]
		if pai >= pai_bar:
			delta = self.winDelta
		else:
			delta = self.loseDelta
		p_move = 0
		comp = {}

		for a in self.possibleActions:
			comp[a] = self.Q[self.logS][a]

		action = [key for key, value in comp.items() if value == max(comp.values())]

		not_star = [x for x in self.possibleActions if x not in action]

		if len(action) == 6:
			for a in action:
				result.append(self.policy[self.logS][a])
			
			return result
		
		for a_nstar in not_star:
			p_move += min(delta/len(not_star),self.policy[self.logS][a_nstar])
			self.policy[self.logS][a_nstar] -= min(delta/len(not_star),self.policy[self.logS][a_nstar])
			
		
		for a_star in action:
			self.policy[self.logS][a_star] += (p_move/len(action))
			
		for a in self.possibleActions:
			result.append(self.policy[self.logS][a])		
		return result

	
	def toStateRepresentation(self, state):
		if state == 'GOAL':
			return ('G',)
		elif state == 'OUT_OF_BOUNDS' or state == 'OUT_OF_TIME':
			return ('O',)
		else:
			return tuple(state[0][0])

	def setState(self, state):
		self.curS = state
		self.C[state] += 1

	def setLearningRate(self,lr):
		self.learningRate = lr
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.learningRate = self.learningRate/(episodeNumber+1)
		return self.loseDelta, self.winDelta, self.learningRate

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			
			observation = nextObservation
