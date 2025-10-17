# PROJECT 1 - The Eight Puzzle

class EightPuzzle:

    def __init__(self, puzz_board, move, depth, cost, parent, initial_state = None):

        # Initialize start state
        if initial_state is None:
            self.state = [[1,2,3],[4,5,6],[7,8,0]]
        else:
            self.state = initial_state

        self.empty_pos = self.find_empty_pos()

        self.puzz_board = puzz_board #current board state
        self.move = move #movement of blank tile (up, down, left, right)
        self.depth = depth #depth of search tree (same as cost for UCS)
        self.cost = cost #depth + heuristic
        self.parent = parent #node's parent state

    def find_empty_pos(self):
        # Find the position of the empty tile (0)
        for i in range(3):
            for j in range(3):
                if self.state[i][j] == 0:
                    return(i,j)

        raise ValueError("No empty tile (0) found in the puzzle.")
    
    def is_goal(self):

        return None

class EightPuzzleSolver:

    @staticmethod
    def uniform_cost_search(puzzle):

        return None
    
    @staticmethod
    def a_star_search_misplaced(puzzle, heuristic):

        return None
    
    @staticmethod
    def a_star_search_manhattan(puzzle, heuristic):
        
        return None

if __name__ == "__main__":


    initial_state = [
        [1, 2, 3],
        [4, 8, 0],
        [7, 6, 5]
    ]