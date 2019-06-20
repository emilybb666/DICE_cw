#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random

class MonteCarloAgent(Agent):
    def __init__(self, discountFactor, epsilon, initVals=0.0):

        super(MonteCarloAgent, self).__init__()
        self.attack = HFOAttackingPlayer()
        # initialise all possible states for the agent
        self.State = [(x,y) for x in range(5) for y in range(6)]
        self.State.append("GOAL")
        self.State.append("OUT_OF_BOUNDS")

        # discount
        self.discountFactor = discountFactor
        # epsilon
        self.epsilon = epsilon
        # variable current state
        self.cur = 0
        # 3 empty lists used to record the episode
        self.logA = []
        self.logR = []
        self.logS = []
        # cumulative reward
        self.G = 0
        

        # Q table is a dict where keys are the "states" and values are another dict. 
        # This inside dict contains "actions" as keys and the values are initilised to 0
        self.Q = {}
        # Returns table is a dict where the key is "state-action" pairs
        self.returns = {}
        for s in self.State:
            self.Q[s] = {}
            for a in self.possibleActions:
                self.Q[s][a] = 0
                self.returns[(s,a)] = []

    def learn(self):
        ll = len(self.logR)
        test = []
        
        for i in range(ll-1,-1,-1):
            exist = 0
            R = self.logR[i]
            
            self.G = self.discountFactor * self.G + R
            for ii in range(i-1,-1,-1):

                if (self.logS[i], self.logA[i]) == (self.logS[ii], self.logA[ii]):
                    exist = 1

            if exist == 0:
            # if (self.logS[i] not in self.logS[:i]) or (self.logA[i] not in self.logA[:i]):
                
                self.returns[(self.logS[i], self.logA[i])].append(self.G)
                self.Q[self.logS[i]][self.logA[i]] = np.mean(self.returns[(self.logS[i], self.logA[i])])
                test.append(self.Q[self.logS[i]][self.logA[i]])
        # print(test)
            # re_result = []
            # re_result.append(test.pop(-1))
        return 1,test[::-1]

            
    def toStateRepresentation(self, state):
        return state[0]
        
    def setExperience(self, state, action, reward, status, nextState):

        self.logS.append(state)
        self.logA.append(action)
        self.logR.append(reward)
        #self.log.append(nextState)

    def setState(self, state):
        self.cur = state

    def reset(self):
        self.G = 0
        self.cur = 0
        self.logS = []
        self.logA = []
        self.logR = []

    def act(self):
        comp = {}
        for s in self.State:
            if s == self.cur:
                for a in self.possibleActions:
                    comp[a] = self.Q[s][a]
            action = [key for key, value in comp.items() if value == max(comp.values())]

        flag = np.random.binomial(1, 1-self.epsilon)
        not_star = [x for x in self.possibleActions if x not in action]

        if flag == 1:
            if len(action) == 5:
                result_action = random.choice(action)
            else:
                result_action = random.choice(not_star)
            
        else:
            result_action = random.choice(action)
        return result_action



    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        self.epsilon = (-1/numEpisodes) * episodeNumber + 1
        return self.epsilon


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args=parser.parse_args()

    #Init Connections to HFO Server
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a Monte-Carlo Agent
    agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
    numEpisodes = args.numEpisodes
    numTakenActions = 0
    # Run training Monte Carlo Method
    for episode in range(numEpisodes):
        agent.reset()
        observation = hfoEnv.reset()
        status = 0

        while status==0:
            epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
            observation = nextObservation

        agent.learn()
