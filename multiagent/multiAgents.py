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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        # compute food score:
        currentFood = currentGameState.getFood()
        closestFood = 0
        foodFn = lambda f: 1.0 / (util.manhattanDistance(newPos, f) + 1)
        foodScores = [foodFn(f) for f in newFood.asList()]
        if foodScores:
            closestFood = min(foodScores)
        avgFoodDist = sum(foodScores) / float(len(foodScores)) if foodScores else 0
        # compute ghost score:
        #closestGhost = 0
        ghostFn = lambda g: util.manhattanDistance(newPos, g.getPosition())
        ghostScores = [ghostFn(g) for g in newGhostStates]
        closestGhost = min(ghostScores)
        # below successfully avoids ghost
        if closestGhost < 2:
            return -1.0 / (closestGhost + 1) + closestFood
        # otherwise, prioritize pellets when it's "safe"
        return closestFood + avgFoodDist + successorGameState.getScore()


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
        def max_value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v, a = float("-inf"), None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                new_value, _ = min_value(successor, agentIndex + 1, depth)
                if new_value > v:
                    v = new_value
                    a = action
            return v, a

        def min_value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v, a = float("inf"), None
            nextIndex = (agentIndex + 1) % state.getNumAgents()
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if nextIndex >= 1:
                    new_value, _ = min_value(successor, nextIndex, depth)
                else:
                    new_value, _ = max_value(successor, nextIndex, depth - 1)
                if new_value < v:
                    v = new_value
                    a = action
            return v, a

        value, bestAction = max_value(gameState, 0, self.depth)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v, a = float("-inf"), None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                new_value, _ = min_value(successor, agentIndex + 1, depth, alpha, beta)
                if new_value > v:
                    v = new_value
                    a = action
                if v > beta:
                    return v, a
                alpha = max(alpha, v)
            return v, a

        def min_value(state, agentIndex, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v, a = float("inf"), None
            nextIndex = (agentIndex + 1) % state.getNumAgents()
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if nextIndex >= 1:
                    new_value, _ = min_value(successor, nextIndex, depth, alpha, beta)
                else:
                    new_value, _ = max_value(successor, nextIndex, depth - 1, alpha, beta)
                if new_value < v:
                    v = new_value
                    a = action
                if v < alpha:
                    return v, a
                beta = min(beta, v)
            return v, a
        value, bestAction = max_value(gameState, 0, self.depth, float("-inf"), float("inf"))
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    import random
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v, a = float("-inf"), None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                new_value, _ = exp_value(successor, agentIndex + 1, depth)
                if new_value > v:
                    v = new_value
                    a = action
            return v, a

        def exp_value(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None
            v, a = 0, None
            nextIndex = (agentIndex + 1) % state.getNumAgents()
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                if nextIndex >= 1:
                    v += exp_value(successor, nextIndex, depth)[0]
                else:
                    v += max_value(successor, nextIndex, depth - 1)[0]
            return v / float(len(legalActions)), a

        value, bestAction = max_value(gameState, 0, self.depth)
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()
    # newFood = successorGameState.getFood()
    # newGhostStates = successorGameState.getGhostStates()
    # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # compute food score:
    currentFood = currentGameState.getFood()
    currentPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodFn = lambda f: 1.0 / (util.manhattanDistance(currentPos, f) + 1)
    foodScores = [foodFn(f) for f in currentFood.asList()]
    closestFood = min(foodScores) if foodScores else 0
    avgFoodDist = sum(foodScores) / float(len(foodScores)) if foodScores else 0
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

# Abbreviation
better = betterEvaluationFunction
