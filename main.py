# PROJECT 1 - The Eight Puzzle
import heapq
import math

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
        goal_state = [[1,2,3],[4,5,6],[7,8,0]]
        return self.state == goal_state
    
    # Override < so that priority queue knows lowest cost is being prioritized
    def __lt__(self, other):
        return self.cost < other.cost

class EightPuzzleSolver:

    # general search algorithm
    @staticmethod
    def search(puzzle, algorithm):
        frontier = []
        heapq.heappush(frontier, puzzle)
        explored = set()
        nodes_expanded = 0
        max_queue_size = 1
        
        # loop through frontier nodes
        while frontier:
            max_queue_size = max(max_queue_size, len(frontier))
            current_node = heapq.heappop(frontier) # pop node w/ lowest cost
            
            # print initial state
            if current_node.state == puzzle.state:
                print(f"\nExpanding state")
                EightPuzzleSolver.print_state(current_node.state)
            
            # check is current node is goal and print out info
            if current_node.is_goal():
                print("Goal!!!")
                print(f"Nodes expanded: {nodes_expanded}")
                print(f"Max queue size: {max_queue_size}")
                print(f"Depth: {current_node.depth}")
                return current_node
            
            state_tuple = EightPuzzleSolver.state_to_tuple(current_node.state)
            # skip over node if it's already been explored (cheapeast path has been found)
            if state_tuple in explored:
                continue

            explored.add(state_tuple)
            
            # print what state is being expanded + cost + depth
            if current_node.state != puzzle.state:
                print(f"\nThe best state to expand with g(n) = {current_node.depth} and h(n) = {current_node.cost - current_node.depth} is...")
                EightPuzzleSolver.print_state(current_node.state)
                print("Expanding this node...\n")

            nodes_expanded += 1
            
            successors = EightPuzzleSolver.get_successors(current_node)
            
            # loop through all successors of node
            for successor in successors:
                successor_tuple = EightPuzzleSolver.state_to_tuple(successor.state)
                if successor_tuple not in explored:
                    # check which search algorithm is being used for heuristic
                    if algorithm == "uniform":
                        h_n = 0
                    elif algorithm == "misplaced":
                        h_n = EightPuzzleSolver.misplaced_heuristic(successor.state)
                    elif algorithm == "euclidean":
                        h_n = EightPuzzleSolver.euclidean_heuristic(successor.state)
                    g_n = successor.depth 
                    successor.cost = g_n + h_n #find cost
                    heapq.heappush(frontier, successor) #push successors into queue
        
        return None

    # Calculate misplaced tile heuristic
    @staticmethod
    def misplaced_heuristic(state):
        heuristic = 0
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        
        for i in range(3):
            for j in range(3):
                # skip blank tile
                if state[i][j] == 0:
                    continue
                # compare state tile w/ goal tile
                if state[i][j] != goal_state[i][j]:
                    heuristic += 1
        return heuristic
    
    @staticmethod
    def euclidean_heuristic(state):
        # dict for goal state
        goal_state = {
            1: (0, 0), 2: (0, 1), 3: (0, 2),
            4: (1, 0), 5: (1, 1), 6: (1, 2),
            7: (2, 0), 8: (2, 1), 0: (2, 2)
        }
        # dict for current state
        state_tile_coords = {}
        total_distance = 0

        # populate current state dict w/ tile # and coordinates
        for i in range(3):
            for j in range(3):
                state_tile = state[i][j]
                state_tile_coords[state_tile] = (i,j)
        
        # calculate euclidean distance between each corresponding tile
        for tile, coord in state_tile_coords.items():
            (i, j) = coord # current coordinate of tile
            (goal_i, goal_j) = goal_state[tile] # find goal coordinate of corresponding tile

            euclidean_distance = math.sqrt(pow((goal_i - i), 2) + pow((goal_j - j), 2))
            total_distance += euclidean_distance

        return total_distance
    
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
    
    # convert state to tuple
    @staticmethod
    def state_to_tuple(state):
        return tuple(tuple(row) for row in state)
    
    # get successors of current node
    @staticmethod
    def get_successors(node):
        successors = []
        empty_pos = node.empty_pos
        
        # all possible moves
        moves = [
            ("Left", EightPuzzleSolver.move_left),
            ("Right", EightPuzzleSolver.move_right),
            ("Up", EightPuzzleSolver.move_up),
            ("Down", EightPuzzleSolver.move_down)
        ]
        
        # loop through all possible moves blank tile can make
        for move_name, move_func in moves:
            new_state = move_func(node.state, empty_pos)
            if new_state is not None: # check legal move
                # create new successor nodes w/ corresponding current node as parent
                successor = EightPuzzle(
                    move=move_name,
                    depth=node.depth + 1,
                    cost=0,
                    parent=node,
                    initial_state=new_state
                )
                successors.append(successor)
        
        return successors
    
    @staticmethod
    def print_state(state):
        for row in state:
            print(' '.join('b' if tile == 0 else str(tile) for tile in row))

if __name__ == "__main__":

    print("Welcome to our 8 puzzle solver!")
    puzzle_select = int(input("Type “1” to use a default puzzle, or “2” to enter your own puzzle.\n"))

    if (puzzle_select == 1):
        board = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

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
    1
    puzz = EightPuzzle(None, 0, 0, None, board)

    algo_select = int(input("Enter your choice of algorithm\n1. Uniform Cost Serch\n2. A* With Misplaced Tile Heuristic\n3. A* With Euclidean Distance Heuristic\n"))
    if algo_select == 1:
        puzz.cost == 0
        EightPuzzleSolver.search(puzz, "uniform")
    elif algo_select == 2:
        EightPuzzleSolver.search(puzz, "misplaced")
    elif algo_select == 3:
        EightPuzzleSolver.search(puzz, "euclidean")
        

