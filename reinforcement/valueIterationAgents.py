# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            keyValue = self.values.copy()
            for state in self.mdp.getStates():
                largest = -9999999999
                if not self.mdp.isTerminal(state):
                    for action in self.mdp.getPossibleActions(state):
                        if self.computeQValueFromValues(state, action) > largest:
                            largest = self.computeQValueFromValues(state, action)
                    keyValue[state] = largest
            self.values = keyValue

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for s, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, s)
            discountedValue = self.discount * self.values[s]
            value = probability * (reward + discountedValue)
            qValue += value
        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        else:
            value = -9999999999
            bestAction = None
            for action in self.mdp.getPossibleActions(state):
                qValue = self.computeQValueFromValues(state, action)
                if qValue > value:
                    value = qValue
                    bestAction = action
            return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        iterations = 0
        while iterations < self.iterations:
            states = self.mdp.getStates()
            s = states[iterations % len(states)]
            if not self.mdp.isTerminal(s):
                actionValues = util.Counter()
                actions = self.mdp.getPossibleActions(s)
                for action in actions:
                    actionValues[action] = self.computeQValueFromValues(s, action)
                maxAction = actionValues.argMax()
                maxValue = actionValues[maxAction]
                self.values[s] = maxValue
            iterations += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Needed major help on this one RIP me.
        predecessors = {}
        priorityQ = util.PriorityQueue()
        states = self.mdp.getStates()
        for s in states:
            predecessors[s] = set()
        for s in states:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                qValues = []
                for action in actions:
                    nextStateAndProbability = self.mdp.getTransitionStatesAndProbs(s, action)
                    for nextState, prob in nextStateAndProbability:
                        if prob > 0:
                            predecessors[nextState].add(s)
                    qValues.append(self.computeQValueFromValues(s, action))
                priorityQ.push(s, -abs(self.values[s] - max(qValues)))
        for i in range(self.iterations):
            if priorityQ.isEmpty():
                break
            else:
                s = priorityQ.pop()
                if not self.mdp.isTerminal(s):
                    actions = self.mdp.getPossibleActions(s)
                    qValues = [self.computeQValueFromValues(s, action) for action in actions]
                    self.values[s] = max(qValues)
                for p in predecessors[s]:
                    actions = self.mdp.getPossibleActions(p)
                    qValues = [self.computeQValueFromValues(p, action) for action in actions]
                    if abs(self.values[p] - max(qValues)) > self.theta:
                        priorityQ.update(p, -abs(self.values[p] - max(qValues)))
