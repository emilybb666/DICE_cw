#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import random
import numpy as np

class SARSAAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(SARSAAgent, self).__init__()

        self.State = [(x,y) for x in range(5) for y in range(6)]
        self.State.append("G")
        self.State.append("O")
        self.discountFactor = discountFactor
        self.learningRate = learningRate
        self.epsilon = epsilon
        # self.epsilon = 0.1
        # 3 empty lists used to record the episode
        self.logA = []
        self.logS = []
        self.R = []
        self.curS = 0
        self.curA = 0

        # Q table is a dict where keys are the "states" and values are another dict.
        # This inside dict contains "actions" as keys and the values are initilised to 0
        self.Q = {}

        for s in self.State:
            self.Q[s] = {}
            for a in self.possibleActions:
                self.Q[s][a] = 0
                # self.returns[(s,a)] = []
        # self.log = {state:[], }


    def learn(self):
        cur_state = self.logS.pop(0)
        cur_action = self.logA.pop(0)
        r = self.R.pop(0)
        before = self.Q[cur_state][cur_action]
        if self.curS == None:
            self.Q[cur_state][cur_action] += self.learningRate * (r- self.Q[cur_state][cur_action])
        else:

            self.Q[cur_state][cur_action] += self.learningRate * (r + self.discountFactor*self.Q[self.logS[-1]][self.logA[-1]] - self.Q[cur_state][cur_action])


        return  self.Q[cur_state][cur_action] - before

    def act(self):
        comp = {}
        for a in self.possibleActions:
            comp[a] = self.Q[self.curS][a]
        action = [key for key, value in comp.items() if value == max(comp.values())]

        if len(action)==5:
            return random.choice(action)


        flag = np.random.binomial(1, 1-self.epsilon)
        not_star = [x for x in self.possibleActions if x not in action]

        if flag == 1:
            result_action = random.choice(action)

        else:
            result_action = random.choice(not_star)
        return result_action

    def setState(self, state):
        self.curS = state

    def setExperience(self, state, action, reward, status, nextState):
        # self.s = state
        # self.a = action
        self.logS.append(state)
        self.logA.append(action)
        # self.logR.append(reward)
        # self.lognexS.append(nextState)
        self.R.append(reward)
        self.curS = nextState
        if nextState != None:

            self.curA = self.act()
        # self.lognexA.append()


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        self.epsilon = (-1/numEpisodes) * episodeNumber + 1
        self.learningRate = self.learningRate/(episodeNumber+1)
        return self.learningRate, self.epsilon

    def toStateRepresentation(self, state):
        return state[0]

    def reset(self):
        self.curS = 0
        self.curA = 0
        self.logA = []
        self.logS = []
        self.R = []


    def setLearningRate(self, learningRate):
        self.learningRate = learningRate


    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args=parser.parse_args()

    numEpisodes = args.numEpisodes
    # Initialize connection to the HFO environment using HFOAttackingPlayer
    hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
    hfoEnv.connectToServer()

    # Initialize a SARSA Agent
    agent = SARSAAgent(0.1, 0.99, 1)

    # Run training using SARSA
    numTakenActions = 0
    for episode in range(numEpisodes):
        agent.reset()
        status = 0

        observation = hfoEnv.reset()
        nextObservation = None
        epsStart = True

        while status==0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            print(obsCopy, action, reward, nextObservation)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))

            if not epsStart :
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
        agent.learn()
