# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
	"""
	  A reflex agent chooses an action at each choice point by examining
	  its alternatives via a state evaluation function.

	  The code below is provided as a guide.  You are welcome to change
	  it in any way you see fit, so long as you don't touch our method
	  headers.
	"""


	def getAction(self, gameState):
		"""
		You do not need to change this method, but you're welcome to.

		getAction chooses among the best options according to the evaluation function.

		Just like in the previous project, getAction takes a GameState and returns
		some Directions.X for some X in the set {North, South, West, East, Stop}
		"""
		# Collect legal moves and successor states
		legalMoves = gameState.getLegalActions()

		# Choose one of the best actions
		scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
		bestScore = max(scores)
		bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
		chosenIndex = random.choice(bestIndices) # Pick randomly among the best

		"Add more of your code here if you want to"

		return legalMoves[chosenIndex]

	def evaluationFunction(self, currentGameState, action):
		"""
		Design a better evaluation function here.

		The evaluation function takes in the current and proposed successor
		GameStates (pacman.py) and returns a number, where higher numbers are better.

		The code below extracts some useful information from the state, like the
		remaining food (newFood) and Pacman position after moving (newPos).
		newScaredTimes holds the number of moves that each ghost will remain
		scared because of Pacman having eaten a power pellet.

		Print out these variables to see what you're getting, then combine them
		to create a masterful evaluation function.
		"""
		# Useful information you can extract from a GameState (pacman.py)
		successorGameState = currentGameState.generatePacmanSuccessor(action)
		newPos = successorGameState.getPacmanPosition()
		newFood = successorGameState.getFood()
		newGhostStates = successorGameState.getGhostStates()
		newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

		"*** YOUR CODE HERE ***"
		manDistFromPacToClosestGhost = 999999999
		for ghostPositions in newGhostStates:
			manDistFromPacToGhost = manhattanDistance(newPos, ghostPositions.getPosition())
			if manDistFromPacToGhost < manDistFromPacToClosestGhost:
				manDistFromPacToClosestGhost = manDistFromPacToGhost
		distanceToFood = 0
		foodList = newFood.asList()
		if foodList:
			manDistFromPosToFood = []
			for food in foodList:
				manDistFromPosToFood.append(manhattanDistance(newPos, food))
			closestFood = min(manDistFromPosToFood)
			distanceToFood = closestFood
		# give values of rewards
		ghostScariness = 0
		foodReward = 0
		movementReward = 0
		if manDistFromPacToClosestGhost > 2:
			ghostScariness = 0
			foodReward = 999999999
			movementReward = 1000
		else:
			ghostScariness = 999999999
			foodReward = 0
			movementReward = 0
		# make it better to move if farther from ghost
		ghostInfluence = ghostScariness * manDistFromPacToClosestGhost
		foodInfluence = foodReward * ((newFood.height * newFood.width) - len(foodList))
		distanceInfluence = movementReward * ((newFood.height + newFood.width) - distanceToFood)
		val = ghostInfluence + foodInfluence + distanceInfluence
		return val


def scoreEvaluationFunction(currentGameState):
	"""
	  This default evaluation function just returns the score of the state.
	  The score is the same one displayed in the Pacman GUI.

	  This evaluation function is meant for use with adversarial search agents
	  (not reflex agents).
	"""
	return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
	"""
	  This class provides some common elements to all of your
	  multi-agent searchers.  Any methods defined here will be available
	  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

	  You *do not* need to make any changes here, but you can if you want to
	  add functionality to all your adversarial search agents.  Please do not
	  remove anything, however.

	  Note: this is an abstract class: one that should not be instantiated.  It's
	  only partially specified, and designed to be extended.  Agent (game.py)
	  is another abstract class.
	"""

	def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
		self.index = 0 # Pacman is always agent index 0
		self.evaluationFunction = util.lookup(evalFn, globals())
		self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent (question 2)
	"""
	def getValue(self, agent, gameState, depth):
		numAgents = gameState.getNumAgents()
		if (depth == 0 or gameState.isLose() or gameState.isWin()):
			return self.evaluationFunction(gameState)
		legalActions = gameState.getLegalActions(agent)
		if Directions.STOP in legalActions:
			legalActions.remove(Directions.STOP)
		successors = []
		for ghostFutureAction in legalActions:
			actions = gameState.generateSuccessor(agent, ghostFutureAction)
			successors.append(actions)
		states = []
		if agent == 0:
			for action in successors:
				states.append(self.getValue(agent + 1, action, depth - 1))
			return max(states)
		else:
			for action in successors:
				states.append(self.getValue((agent + 1) % numAgents, action, depth - 1))
			return min(states)

	def getAction(self, gameState):
		"""
		  Returns the minimax action from the current gameState using self.depth
		  and self.evaluationFunction.

		  Here are some method calls that might be useful when implementing minimax.

		  gameState.getLegalActions(agentIndex):
			Returns a list of legal actions for an agent
			agentIndex=0 means Pacman, ghosts are >= 1

		  gameState.generateSuccessor(agentIndex, action):
			Returns the successor game state after an agent takes an action

		  gameState.getNumAgents():
			Returns the total number of agents in the game

		  gameState.isWin():
			Returns whether or not the game state is a winning state

		  gameState.isLose():
			Returns whether or not the game state is a losing state
		"""
		"*** YOUR CODE HERE ***"
		trueDepth = gameState.getNumAgents() * self.depth
		# get legal actions of pacman
		legalActions = gameState.getLegalActions(0)
		if Directions.STOP in legalActions:
			legalActions.remove(Directions.STOP)
		# list of future pacman action possiblities
		successors = []
		for pacFutureAction in legalActions:
			actions = gameState.generateSuccessor(0, pacFutureAction)
			successors.append(actions)
		# get the value for each successor
		values = []
		for action in successors:
			values.append(self.getValue(1, action, trueDepth - 1))
		biggestValue = max(values)
		choices = []
		x = 0
		while x < len(values):
			if biggestValue == values[x]:
				choices.append(x)
			x += 1
		return legalActions[random.choice(choices)]

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	  Your minimax agent with alpha-beta pruning (question 3)
	"""
	def getValue(self, agent, gameState, depth, a, b):
		numAgents = gameState.getNumAgents()
		if (depth == 0 or gameState.isLose() or gameState.isWin()):
			return self.evaluationFunction(gameState)
		legalActions = gameState.getLegalActions(agent)
		if Directions.STOP in legalActions:
			legalActions.remove(Directions.STOP)
		# get the maximum
		if agent == 0:
			value = -999999999
			for legalAction in legalActions:
				successor = gameState.generateSuccessor(agent, legalAction)
				val = self.getValue(agent + 1, successor, depth - 1, a, b)
				value = max(value, val)
				if value > b:
					return value
				if a < value:
					a = value
			return value
		# get the minimum
		else:
			value = 999999999
			for legalAction in legalActions:
				successor = gameState.generateSuccessor(agent, legalAction)
				val = self.getValue((agent + 1) % numAgents, successor, depth - 1, a, b)
				value = min(value, val)
				if value < a:
					return value
				if b > value:
					b = value
			return value

	def getAction(self, gameState):
		"""
		  Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"
		alpha = -999999999
		beta = 999999999
		trueDepth = gameState.getNumAgents() * self.depth
		if (trueDepth == 0 or gameState.isLose() or gameState.isWin()):
			return self.evaluationFunction(gameState)
		legalActions = gameState.getLegalActions(0)
		if Directions.STOP in legalActions:
			legalActions.remove(Directions.STOP)
		bestVal = -999999999
		bestLegalAction = None
		# get best action
		for state in legalActions:
			successor = gameState.generateSuccessor(0, state)
			value = self.getValue(1, successor, trueDepth - 1, alpha, beta)
			if value > bestVal:
				bestVal = value
				bestLegalAction = state
			if alpha < value:
				alpha = value
		return bestLegalAction

class ExpectimaxAgent(MultiAgentSearchAgent):
	"""
	  Your expectimax agent (question 4)
	"""
	def getValue(self, agent, gameState, depth):
		numAgents = gameState.getNumAgents()
		if (depth == 0 or gameState.isLose() or gameState.isWin()):
			return self.evaluationFunction(gameState)
		legalActions = gameState.getLegalActions(agent)
		if Directions.STOP in legalActions:
			legalActions.remove(Directions.STOP)
		succ = []
		for actions in legalActions:
			action = gameState.generateSuccessor(agent, actions)
			succ.append(action)
		# maximum
		if agent == 0:
			maximum = -999999999
			for action in succ:
				value = self.getValue(agent + 1, action, depth - 1)
				if value > maximum:
					maximum = value
			return maximum
		# expected
		else:
			sumOfValues = 0
			count = 0
			for action in succ:
				value = self.getValue((agent + 1) % numAgents, action, depth - 1)
				sumOfValues += value
				count += 1
			return float(sumOfValues / count)

	def getAction(self, gameState):
		"""
		  Returns the expectimax action using self.depth and self.evaluationFunction

		  All ghosts should be modeled as choosing uniformly at random from their
		  legal moves.
		"""
		"*** YOUR CODE HERE ***"
		trueDepth = gameState.getNumAgents() * self.depth
		legalActions = gameState.getLegalActions(0)
		if Directions.STOP in legalActions:
			legalActions.remove(Directions.STOP)
		successors = []
		for actions in legalActions:
			action = gameState.generateSuccessor(0, actions)
			successors.append(action)
		values = []
		for state in successors:
			val = self.getValue(1, state, trueDepth - 1)
			values.append(val)
		maxValue = max(values)
		maximum = []
		x = 0
		while x < len(values):
			if maxValue == values[x]:
				maximum.append(x)
			x += 1
		return legalActions[random.choice(maximum)]

def betterEvaluationFunction(currentGameState):
	"""
	  Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	  evaluation function (question 5).

	  DESCRIPTION: <write something here so we know what you did>
	  My plan is to give winning the best value so I set it to an extremely high number if pacman wins.
	  If pacman loses, then the score if going to be really bad.
	  If pacman is close to a ghost, then pacman will run away.
	  This is done by penalizing pacman when it becomes too close to a ghost.
	  Because I wanted survivability, I made food worth less than the penalty for pacman being too close to a ghost.
	  I also made food that are farther away worth less than the ones closer to pacman to encourage pacman to get the food pellets that are closest.
	  I also wanted to give pacman a bonus for eating more food than not to eat any.
	  If pacman was near a power pellet, then pacman will want to get it only if it is near it.
	"""
	"*** YOUR CODE HERE ***"
	if currentGameState.isWin():
		return 999999999
	if currentGameState.isLose():
		return -999999999
	score = scoreEvaluationFunction(currentGameState)
	foodLocations = currentGameState.getFood().asList()
	closestFood = 999999999
	# find manhattan distance from food positions and pacman position and get the closest food pellet
	for foodPosition in foodLocations:
		distance = util.manhattanDistance(foodPosition, currentGameState.getPacmanPosition())
		if (distance < closestFood):
			closestFood = distance
	i = 1
	closestGhostDistance = 999999999
	# find manhattan distance from pacman position and all other ghosts
	while i < currentGameState.getNumAgents():
		ghostDistances = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(i))
		closestGhostDistance = min(closestGhostDistance, ghostDistances)
		i += 1
	# penalize pacman even more than default penalty for staying live when pacman gets too close to ghost
	if closestGhostDistance > 4:
		score += closestGhostDistance * 2
	# default penalty for staying alive; encourages pacman to seek food instead of idling
	else:
		score += 8
	# make pacman want to get the closest food pellet
	score -= closestFood * 1.5
	# give pacman incentive to eat food in general
	score -= 4 * len(foodLocations)
	# give pacman incentive to eat power pellet only if close
	powerPellets = currentGameState.getCapsules()
	score -= 3.5 * len(powerPellets)
	return score


# Abbreviation
better = betterEvaluationFunction

