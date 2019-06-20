from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self, discount = 0.9, theta = 1e-4):
		self.MDP = MDP()
		self.discount = discount
		self.theta = theta
		self.initval, self.policy = self.initVs()


	def initVs(self):
		initval = {}
		policy = {}
		L1 = self.MDP.S
		for i in L1:
			initval[i] = 0
			# all the action
			policy[i] = self.MDP.A

		return initval, policy


	def BellmanUpdate(self):
		for states in self.MDP.S:
			nextV = {}
			for action in self.MDP.A:
				nextStateProb = self.MDP.probNextStates(states, action)
				
				value = 0
				for nextsta in nextStateProb:
					immr = self.MDP.getRewards(states, action, nextsta)
					
					value += nextStateProb[nextsta] * (immr + self.discount * self.initval[nextsta])
					
				nextV[action] = value
	
			self.initval[states] = max(nextV.values())
			# select the corresponding optimal action and fill in the policy dic
			self.policy[states] = [key for key, value in nextV.items() if value == max(nextV.values())]			



		return self.initval, self.policy			
			
		
	

if __name__ == '__main__':
	solution = BellmanDPSolver()
	#solution.initVs()

	for i in range(20000):
		values, policy = solution.BellmanUpdate()

	print("Values : ", values)
	print('\n')
	print("Policy : ", policy)