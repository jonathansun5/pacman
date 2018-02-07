# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack()
    visited_nodes = set()
    fringe.push((problem.getStartState(), []))
    while not fringe.isEmpty():
        current = fringe.pop()
        state = current[0]
        path = current[1]
        if problem.isGoalState(state):
            return path
        if state not in visited_nodes:
            visited_nodes.add(state)
            for successor in problem.getSuccessors(state):
                suc_state = successor[0]
                suc_path = successor[1]
                fringe.push((suc_state, path + [suc_path]))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    visited_nodes = set()
    fringe.push((problem.getStartState(), []))
    while not fringe.isEmpty():
        current = fringe.pop()
        state = current[0]
        path = current[1]
        if problem.isGoalState(state):
            return path
        if state not in visited_nodes:
            visited_nodes.add(state)
            for successor in problem.getSuccessors(state):
                suc_state = successor[0]
                suc_path = successor[1]
                fringe.push((suc_state, path + [suc_path]))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    visited_nodes = set()
    fringe.push((problem.getStartState(), 0, []), 0)
    while not fringe.isEmpty():
        current = fringe.pop()
        state = current[0]
        cost = current[1]
        path = current[2]
        if problem.isGoalState(state):
            return path
        if state not in visited_nodes:
            visited_nodes.add(state)
            for successor in problem.getSuccessors(state):
                suc_state = successor[0]
                suc_path = successor[1]
                suc_cost = successor[2]
                fringe.push((suc_state, suc_cost + cost, path + [suc_path]), suc_cost + cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def getListOfActions(path):
    pathList = []
    for arc in path:
        if arc[1] is not "":
            pathList.append(arc[1])
    return pathList

def findSolution(problem=None, startNode=(((0, 0), "", 0)), dataStructure=util.Stack(), closedSet=None):
    """
    A function that takes a problem and identifies if there is a solution to the pacman maze.  Returns
    a list of arcs if the solution does exist.
    """
    nodeLocationIndex = 0
    nodeArcDirectionIndex = 1
    nodeArcCostIndex = 2
    problemStateIndex = 3
    if problem is None:
        return None
    if dataStructure.isEmpty():
        return None
    while not dataStructure.isEmpty():
        destPath = dataStructure.pop()
        destNode = destPath[-1]
        destNodeCord = destNode[nodeLocationIndex]
        consideredNodeDir = destNode[nodeArcDirectionIndex]
        problemState = None
        if closedSet is not None and destNodeCord in closedSet:
            continue
        if problemState is not None and problem.isGoalState(problemState):
            return destPath
        elif problemState is None and problem.isGoalState(destNodeCord):
            return destPath
        successors = ()
        successors = problem.getSuccessors(destNodeCord)
        if not successors:
            continue
        nodesThisLevel = len(successors)
        for node in successors:
            dataStructure.push(tuple(list(destPath) + [node]))
        if closedSet is not None:
            closedSet.add(destNodeCord)
    return None

def getFn(gN, hN):
    return gN + hN

def getHeuristicFunction(problem, heuristic):
    return lambda (path): getFn(problem.getCostOfActions(getListOfActions(path)), heuristic(path[-1][0], problem))

def aStarSearch(problem, heuristic=nullHeuristic):
    closedSet = set()
    dataStructure = util.PriorityQueueWithFunction(getHeuristicFunction(problem, heuristic))
    path = []
    pathTuple = ()
    if "startState" in dir(problem):
        nodeCoordStartState = problem.startState
        pathTuple = ((nodeCoordStartState, "", 0),)
    elif "getStartState" in dir(problem):
        nodeCoordStartState = problem.getStartState()
        pathTuple = ((nodeCoordStartState, "", 0),)
    else:
        raise Exception("No recognizable function for getting the Start State")
    dataStructure.push(pathTuple)
    result = findSolution(problem, pathTuple, dataStructure, closedSet)
    if result is None:
        raise Exception("No solution exists!")
    path = getListOfActions(result)
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
