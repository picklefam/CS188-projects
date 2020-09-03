# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """
    def evaluationFunction(self, currentGameState):
        currentFood = currentGameState.getFood()
        currentPosition = currentGameState.getPacmanPosition(self.index)
        ghostStates = currentGameState.getGhostStates()
        scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

        foodFn = lambda f: 1.0 / (util.manhattanDistance(currentPos, f) + 1)
        foodScores = [foodFn(f) for f in currentFood.asList()]
        closestFood = min(foodScores) if foodScores else 0
        avgFoodDist = sum(foodScores) / float(len(foodScores)) if foodScores else 0
        return avgFoodDist

        end = min(4, len(foodScores))
        foodScores = sorted(foodScores, reverse=True)
        closest3 = sum(foodScores[0:end])
        # compute ghost score:
        ghostFn = lambda g: util.manhattanDistance(currentPos, g.getPosition())
        ghostScores = [ghostFn(g) for g in ghostStates]
        closestGhost = min(ghostScores) if ghostScores else 0

        # annihilate the ghosts!!
        if all(t > 1 for t in scaredTimes):
            return 1 / (closestGhost + 1) + closest3 + avgFoodDist + currentGameState.getScore()
        # below successfully avoids ghost
        if closestGhost < 2:
            return -1.0 / (closestGhost + 1) + closest3 + currentGameState.getScore()
        # otherwise, ignore ghosts and nab pellets
        return closest3 + avgFoodDist + currentGameState.getScore()


    def getAction(self, gameState):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalPacmanActions(self.index)
        currentPos = gameState.getPacmanPosition(self.index)
        otherPacmen = gameState.getPacmanPositions()
        otherPacmen.remove(currentPos)
        print(otherPacmen)
        print(gameState.getNumPacmanAgents())
        print(currentPos, legalMoves)
        proposedStates = [gameState.generatePacmanSuccessor(action, self.index) for action in legalMoves]
        # Choose one of the best actions
        for nextState in proposedStates:
            if nextState.getPacmanPosition(self.index) in otherPacmen:
                proposedStates.remove(nextState)
        for p in proposedStates:
            print(p.getPacmanPosition(self.index))
        # return first action to see if it's working
        scores = [self.evaluationFunction(state) for state in proposedStates]
        print("evaluated")
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"
        proposedState = gameState.generatePacmanSuccessor(legalMoves[chosenIndex], self.index)
        print(proposedState.getPacmanPosition(self.index))
        print(legalMoves[chosenIndex])
        return legalMoves[chosenIndex]

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"

        #raise NotImplementedError()

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"
        return (search.uniformCostSearch(problem))

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        return (x,y) in self.food.asList()
