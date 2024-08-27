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

        "*** YOUR CODE HERE ***" 
       #Evaluate the ghost distances and assign a score based on distance and ghost scared times
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghost_score = 0
        for ghost, distance in zip(newGhostStates, ghost_distances):
            if ghost.scaredTimer > distance:
                ghost_score += float('inf')
            elif distance <= 1:
                ghost_score += float('-inf')
                
        # Evaluate the distance to the closest food pellet and assign a score based on the inverse of the distance
        food_distances = [manhattanDistance(newPos, food_pos) for food_pos in newFood.asList()]
        closest_food_distance = min(food_distances, default=float('inf'))
        closest_food_feature = 1.0 / (1.0 + closest_food_distance)

        # Return the sum of the scores for the ghost proximity, food proximity, and the game score
        return successorGameState.getScore() + ghost_score + closest_food_feature

    



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
        def minimax(state, depth, agent):
            '''
                Returns the best value-action pair for the agent
            '''


            # Determine the next depth based on the current agent

            min_Depth = depth-1 if agent == 0 else depth
            # If we've reached the end of the tree or the game is over, return the evaluation function score
            if min_Depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Set the best option and initial value depending on whether it's a maximizing or minimizing agent
            best_option, bestVal = (max, -float('inf')) if agent == 0 else (min, float ('inf'))
            # Get the next agent in line
            next_Agent = (agent + 1) % state.getNumAgents()
            
            # Loop over all legal actions for the current agent
            bestAction = None
            for action in state.getLegalActions(agent):
                # Generate the successor state
                successorState = state.generateSuccessor(agent, action)
                # Call minimax recursively on the successor state
                valOfAction, _ = minimax(successorState, min_Depth, next_Agent)
                
                # Check whether the value of the current action is better than the previous best value
                if best_option(bestVal, valOfAction) == valOfAction:
                    bestVal = valOfAction
                    bestAction = action
            return bestVal, bestAction
        
        # Call the minimax function on the current game state, and return the best action
        val, action = minimax(gameState, self.depth+1, self.index)
        return action

  


        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def alphaBeta(state, state_depth, alpha, beta, agent):
            # Determine if the current player is the maximizing player (agent 0)
            alph_Max = agent == 0
            # Determine the next depth to search
            next_depth = state_depth - 1 if alph_Max else state_depth
            
            # If we've reached the maximum depth or the game has ended, return the evaluation of the node and None as the best action
            if next_depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Determine the next player
            nextAgent = (agent + 1) % state.getNumAgents()
            
            # Initialize the best value and best action seen so far
            val = -float('inf') if alph_Max else float('inf')
            agent_action = None
            
            # Determine whether we are maximizing or minimizing
            best_option = max if alph_Max else min
            
            # Loop over all possible actions
            for action in state.getLegalActions(agent):
                # Generate the successor state for the current action
                successorState = state.generateSuccessor(agent, action)
                # Recursively call the alphaBeta function on the successor state
                valOfAction, _ = alphaBeta(successorState, next_depth, alpha, beta, nextAgent)
                
                # If the value of the current action is better than the best value seen so far, update the best value and best action
                if best_option(val, valOfAction) == valOfAction:
                    val, agent_action = valOfAction, action
                
                # Update alpha and beta values if necessary
                if alph_Max:
                    if val > beta:
                        return val, agent_action
                    alpha = max(alpha, val)
                else:
                    if val < alpha:
                        return val, agent_action
                    beta = min(beta, val)
            return val, agent_action

        # Call the alphaBeta function on the initial game state with the initial alpha and beta values and the index of the current player
        _, action = alphaBeta(gameState, self.depth+1, -float('inf'), float('inf'), self.index)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        expect_agent = self.index
        if expect_agent != 0:
            return random.choice(state.getLegalActions(expect_agent))

        def expectimax(state, expect_depth, agent_exp):
            '''
            Returns the best value-action pair for the agent
            '''
            # Determine the next depth to search
            nextDepth = expect_depth-1 if agent_exp == 0 else expect_depth
            
            # If we've reached the maximum depth or the game has ended, return the evaluation of the node and None as the best action
            if nextDepth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Determine the next player
            nextAgent = (agent_exp + 1) % state.getNumAgents()
            # Get the legal moves for the current player
            legalMoves = state.getLegalActions(agent_exp)
            
            # If the current player is not the maximizing player, calculate the expected value of each action
            if agent_exp != 0:
                prob = 1.0 / float(len(legalMoves))
                value = 0.0
                for action in legalMoves:
                    # Generate the successor state for the current action
                    successorState = state.generateSuccessor(agent_exp, action)
                    # Recursively call the expectimax function on the successor state
                    expVal, _ = expectimax(successorState, nextDepth, nextAgent)
                    # Update the expected value
                    value += prob * expVal
                return value, None
    
            # Otherwise, find the action that maximizes the expected value
            bestVal, bestAction = -float('inf'), None
            for action in legalMoves:
                # Generate the successor state for the current action
                successorState = state.generateSuccessor(agent_exp, action)
                # Recursively call the expectimax function on the successor state
                expVal, _ = expectimax(successorState, nextDepth, nextAgent)
                # Update the best value and best action seen so far
                if max(bestVal, expVal) == expVal:
                    bestVal, bestAction = expVal, action
            return bestVal, bestAction


        #Call the expectimax function on the initial game state with the initial depth and index of the current player
        _, action = expectimax(gameState, self.depth+1, self.index)
        return action
foodSearch = None

#util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #util.raiseNotDefined()

    # Extract relevant information from the game state
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    #distToPacman = partial(var1, currentGameState, pos)
    capsuleGrid = currentGameState.getCapsules()
    
    # Initialize variables to store distances and score
    foodDistances = []
    ghostDistances = []
    score = currentGameState.getScore()
    
    # Evaluate distance to the closest food pellet
    for fd in food.asList():
        foodDistances.append(manhattanDistance(pos, fd))
    if len(foodDistances) == 0:
        return score
    minFoodDistance = min(foodDistances)
    score += 10.0 / minFoodDistance
    
    # Evaluate distance to the closest capsule
    for capsule in capsuleGrid:
        capsuleDistance = manhattanDistance(pos, capsule)
        if capsuleDistance == 0:
            score += 50
        else:
            score += 1.0 / capsuleDistance
            
    # Evaluate distance to the closest ghost
    for i, ghostState in enumerate(ghostStates):
        ghostPosition = ghostState.getPosition()
        ghostDistance = manhattanDistance(pos, ghostPosition)
        if scaredTimes[i] > 0:
            score += 10.0 / (ghostDistance + 1)
        else:
            if ghostDistance <= 1:
                score -= 100
            else:
                ghostDistances.append(ghostDistance)
    if len(ghostDistances) > 0:
        minGhostDistance = min(ghostDistances)
        score -= 5.0 / minGhostDistance
        
    return score


# Abbreviation
better = betterEvaluationFunction


