# PROJECT 1 - The Eight Puzzle

class EightPuzzle:

    def __init__(self, initial_state = None):

        # Initialize start state
        if initial_state is None:
            self.state = [[1,2,3],[4,5,6],[7,8,0]]
        else:
            self.state = initial_state

        self.empty_pos = self.find_empty_pos()

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

class EightPuzzleSolver:

    @staticmethod
    def uniform_cost_search(puzzle):

        return None
    
    @staticmethod
    def a_star_missplaced(puzzle):

        return None
    
    @staticmethod
    def a_star_euclidean(puzzle):

        return None

if __name__ == "__main__":

    print("Welcome to our 8 puzzle solver!")
    puzzle_select = input("Type “1” to use a default puzzle, or “2” to enter your own puzzle.")

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
        

