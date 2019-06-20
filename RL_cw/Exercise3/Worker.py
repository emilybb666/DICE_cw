import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np
import os, sys

def train(idx, args, value_network, target_value_network, optimizer, lock, counter):
	
	port = args.port 
	seed = args.seed
	hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
	hfoEnv.connectToServer()

	eps = args.epsilon
	I_target = args.I_target
	I_update = args.I_update
	discountFactor = args.discountFactor
	learn_step_counter = 0
	loss_func = nn.MSELoss()
	threads = args.numprocesses

# This runs a random agent
	s = hfoEnv.reset()
	while learn_step_counter < counter.value / threads:
		lock.acquire()
		counter.value += 1
		lock.release()
		learn_step_counter += 1
		# take action
		action = choose_action(s, value_network, eps)
		# nextState, reward, done, status, info based on the action
		s_, r, done, status, info = hfoEnv.step(action)
		
		act_index = hfoEnv.possibleActions.index(action)
		q_eval = computePrediction(s, act_index, value_network)
		q_target = computeTargets(r, s_, discountFactor, done, target_value_network)
		loss = loss_func(q_eval, q_target)

		
		loss.backward()
		if learn_step_counter % I_update or done:
			#lock.acquire()
			optimizer.step()
			optimizer.zero_grad()
			optimizer.share_memory()
			#lock.release()
			
		# target parameter update
		if counter.value % I_target == 0:
			lock.acquire()
			target_value_network.load_state_dict(value_network.state_dict())
			lock.release()
		
		if counter.value == 1e6:
			saveModelNetwork(target_value_network, os.getcwd())



		if done:
			s = hfoEnv.reset()

		s = s_

	
	
def choose_action(self, x, value_network, eps):
	possible_action = ["MOVE", "SHOOT", "DRIBBLE", "GO_TO_BALL"]
	x = torch.unsqueeze(torch.FloatTensor(x), 0)
	# input only one sample
	if np.random.uniform() < eps:   # greedy
		actions_value = self.value_network.forward(x)
		action_idx = torch.max(actions_value, 0)[1].data.numpy()
		action = possible_action[action_idx]
		return action

	else:   # random
		action_idx = np.random.randint(0, 4)
		action = possible_action[action_idx]

		return action




def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	q_next = targetNetwork(nextObservation).detach()     # detach from graph, don't backpropagate
	q_target = reward + discountFactor * q_next.max(1)[0]  
	if done:
		q_target = torch.tensor(reward)
	return q_target
def computePrediction(state, action, valueNetwork):
	q_eval = valueNetwork(state)  # shape (batch, 1)
	return q_eval[0][action]
# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
	torch.save(model.state_dict(), strDirectory)




