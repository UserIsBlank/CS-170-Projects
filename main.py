# PROJECT 1 - The Eight Puzzle

class EightPuzzle:

    def __init__(self, move, depth, cost, parent, initial_state = None):

        # Initialize start state
        if initial_state is None:
            self.state = [[1,2,3],[4,5,6],[7,8,0]]
        else:
            self.state = initial_state

        self.empty_pos = self.find_empty_pos()

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
    
    # Override < so that priority queue knows lowest cost is being prioritized
    def __lt__(self, other):
        return self.cost < other.cost

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
    
    # Create new board state after moving empty tile left
    @staticmethod
    def move_left(state, empty_pos):
        new_state = [x[:] for x in state] #make copy of current board state
        i, j = empty_pos

        if j > 0: #make sure empty tile isn't at leftmost column
            # Swap empty tile w/ tile on its left
            new_state[i][j], new_state[i][j - 1] = new_state[i][j - 1], new_state[i][j]
            return new_state
        else:
            return None
        
    # Create new board state after moving empty tile right
    @staticmethod
    def move_right(state, empty_pos):
        new_state = [x[:] for x in state] #make copy of current board state
        i, j = empty_pos

        if j < 2: #make sure empty tile isn't at rightmost column
            # Swap empty tile w/ tile on its right
            new_state[i][j], new_state[i][j + 1] = new_state[i][j + 1], new_state[i][j]
            return new_state
        else:
            return None

    # Create new board state after moving empty tile up
    @staticmethod
    def move_up(state, empty_pos):
        new_state = [x[:] for x in state] #make copy of current board state
        i, j = empty_pos

        if i > 0: #make sure empty tile isn't at topmost row
            # Swap empty tile w/ tile above
            new_state[i][j], new_state[i - 1][j] = new_state[i - 1][j], new_state[i][j]
            return new_state
        else:
            return None
        
    # Create new board state after moving empty tile down
    @staticmethod
    def move_down(state, empty_pos):
        new_state = [x[:] for x in state] #make copy of current board state
        i, j = empty_pos

        if i < 2: #make sure empty tile isn't at bottom-most row
            # Swap empty tile w/ tile below
            new_state[i][j], new_state[i + 1][j] = new_state[i + 1][j], new_state[i][j]
            return new_state
        else:
            return None

if __name__ == "__main__":


    initial_state = [
        [1, 2, 3],
        [4, 8, 0],
        [7, 6, 5]
    ]