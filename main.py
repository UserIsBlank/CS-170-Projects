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

        # Check if current state is the goal state


        return None
    
    # Override < so that priority queue knows lowest cost is being prioritized
    def __lt__(self, other):
        return self.cost < other.cost

class EightPuzzleSolver:

    @staticmethod
    def uniform_cost_search(puzzle):

        return None
    
    @staticmethod
    def a_star_misplaced(puzzle):

        return None
    
    # Calculate misplaced tile heuristic
    @staticmethod
    def misplaced_heuristic(state):
        goal_counter = 0 #goal state is 123456780
        heuristic = 0
        for i in range(3):
            for j in range(3):
                if (i, j) == (2, 2): #check last tile
                    if state[i][j] != 0: 
                        heuristic += 1
                    break
                goal_counter += 1 #increment because goal state is in order
                if state[i][j] != goal_counter: #compare each tile up to last tile
                    heuristic += 1
        return heuristic
    
    @staticmethod
    def a_star_euclidean(puzzle):

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

    print("Welcome to our 8 puzzle solver!")
    puzzle_select = int(input("Type “1” to use a default puzzle, or “2” to enter your own puzzle."))

    if (puzzle_select == 1):
        board = [[4, 6, 8], [2, 0, 1], [5, 3, 7]]

    if (puzzle_select == 2):
        board = []

        print("Enter your puzzle, use a zero to represent the blank")
        print("Enter the first row, use space or tabs between numbers")
        row1 = list(map(int, input().split()))

        print("Enter the second row, use space or tabs between numbers")
        row2 = list(map(int, input().split()))

        print("Enter the third row, use space or tabs between numbers")
        row3 = list(map(int, input().split()))

        board = [row1, row2, row3]
        

