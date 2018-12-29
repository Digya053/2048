from BaseAI_3 import BaseAI
from MinimaxWithPruning_3 import *
import numpy as np


class PlayerAI(BaseAI):

    def getMove(self, grid):
        moves = grid.getAvailableMoves()
        max_utility = -np.inf
        nxt_move = -1

        for move in moves:
            child = get_child(grid, move)

            utility = decision(child, False)

            if utility >= max_utility:
                max_utility = utility
                nxt_move = move

        return nxt_move