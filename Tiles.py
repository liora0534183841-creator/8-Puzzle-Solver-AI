"""
Name: Liora Levi
ID: 326037488
Assignment: 8-Puzzle Solver (BFS & A*)
"""

import sys
import heapq
import numpy as np

# --- Global Constants ---

# The target configuration of the board (0 represents the empty tile)
TARGET_STATE = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint8).reshape(3, 3)

# Movement vectors mapping actions to (row_change, col_change)
MOVE_VECTORS = {
    "Left": (0, -1),
    "Right": (0, 1),
    "Up": (-1, 0),
    "Down": (1, 0)
}


class PuzzleState:
    """
    Represents a single state in the 8-Puzzle game.
    Manages the board configuration, parent tracking for path reconstruction,
    and transition logic.
    """

    def __init__(self, board, parent=None, move_made=None, g_cost=0):
        self.board = board
        self.parent = parent
        self.move_made = move_made
        self.g_cost = g_cost
        self.h_cost = 0  # Will be calculated and updated by calculateHeuristic()

        # Optimization: Cache the zero position to avoid searching the array in every step
        self.zero_pos = tuple(np.argwhere(board == 0)[0])

    def __eq__(self, other):
        """
        Checks if two states are equal based on the board configuration.
        Required for checking if a state exists in the 'visited' set.
        """
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        """
        Generates a unique hash for the board.
        Required to store PuzzleState objects in a set or dictionary.
        """
        return hash(self.board.tobytes())

    def __lt__(self, other):
        """
        Comparison operator for Priority Queue (used in A*).
        Compares states based on total cost f(n) = g(n) + h(n).
        """
        return (self.g_cost + self.h_cost) < (other.g_cost + other.h_cost)

    def isGoal(self):
        """Returns True if the current board matches the target state."""
        return np.array_equal(self.board, TARGET_STATE)

    def getPossibleActions(self):
        """
        Returns a list of valid actions ('Left', 'Right', 'Up', 'Down')
        based on the current position of the empty tile (0).
        """
        actions = []
        row, col = self.zero_pos

        # Check boundaries before allowing a move
        if col > 0: actions.append("Left")
        if col < 2: actions.append("Right")
        if row > 0: actions.append("Up")
        if row < 2: actions.append("Down")

        return actions

    def applyAction(self, action):
        """
        Transition Model: Applies an action and returns a NEW PuzzleState object.
        Uses a deep copy of the board to ensure the current state remains immutable.
        """
        new_board = self.board.copy()
        row, col = self.zero_pos
        delta_row, delta_col = MOVE_VECTORS[action]

        new_row, new_col = row + delta_row, col + delta_col

        # Swap the empty tile (0) with the target tile
        target_val = new_board[new_row, new_col]
        new_board[new_row, new_col] = 0
        new_board[row, col] = target_val

        return PuzzleState(new_board, parent=self, move_made=action, g_cost=self.g_cost + 1)

    def calculateHeuristic(self):
        """
        Calculates the heuristic value (h) using Manhattan Distance + Linear Conflict.
        Updates the self.h_cost attribute.
        """
        self.h_cost = self._manhattan_distance() + (2 * self._linear_conflict())
        return self.h_cost

    def _manhattan_distance(self):
        """
        Calculates the sum of Manhattan distances for all tiles (except 0)
        from their current position to their goal position.
        """
        distance = 0
        for row in range(3):
            for col in range(3):
                value = self.board[row, col]
                if value != 0:
                    # Calculate target position for the current value
                    target_row = value // 3
                    target_col = value % 3
                    distance += abs(row - target_row) + abs(col - target_col)
        return distance

    def _linear_conflict(self):
        """
        Calculates Linear Conflict penalty.
        Adds 2 to the cost for each pair of tiles that are in their correct row/column
        but are in reversed order (blocking each other).
        """
        conflicts = 0

        # Check for row conflicts
        for row in range(3):
            row_values = self.board[row, :]
            for i in range(3):
                for j in range(i + 1, 3):
                    tile_a = row_values[i]
                    tile_b = row_values[j]
                    if tile_a != 0 and tile_b != 0:
                        # If both tiles belong in this row
                        if (tile_a // 3) == row and (tile_b // 3) == row:
                            # If they are in reversed order (A > B means A is to the left of B)
                            if tile_a > tile_b:
                                conflicts += 1

        # Check for column conflicts
        for col in range(3):
            col_values = self.board[:, col]
            for i in range(3):
                for j in range(i + 1, 3):
                    tile_a = col_values[i]
                    tile_b = col_values[j]
                    if tile_a != 0 and tile_b != 0:
                        # If both tiles belong in this column
                        if (tile_a % 3) == col and (tile_b % 3) == col:
                            # If they are in reversed order
                            if tile_a > tile_b:
                                conflicts += 1
        return conflicts


# --- Solver Algorithms ---

def get_solution_path(last_state):
    """
    Reconstructs the solution path by backtracking from the goal state
    to the start state using the parent pointers.
    """
    path = []
    current = last_state
    while current.parent is not None:
        # Identify which tile moved into the empty space
        prev_zero_pos = current.parent.zero_pos
        tile_moved = current.board[prev_zero_pos]
        path.append(tile_moved)
        current = current.parent
    return path[::-1]  # Reverse to get the path from Start -> Goal


def BFS(start_state):
    """
    Breadth-First Search (Un-Informed Search).
    Uses a FIFO queue.
    Note: Includes 'Early Goal Test' optimization (checking goal before adding to queue)
    to match the expansion count reference in the course book.
    """
    if start_state.isGoal():
        return start_state, 0

    queue = [start_state]
    visited = set()
    visited.add(start_state)
    expanded_nodes = 0

    while queue:
        current_state = queue.pop(0)
        expanded_nodes += 1

        for action in current_state.getPossibleActions():
            child_state = current_state.applyAction(action)

            if child_state not in visited:
                # Optimization: Check for goal immediately upon generation
                if child_state.isGoal():
                    return child_state, expanded_nodes
                visited.add(child_state)
                queue.append(child_state)

    return None, expanded_nodes


def A_Star(start_state):
    """
    A* Search (Informed Search).
    Uses a Priority Queue (Min-Heap) to expand nodes with the lowest f-cost (g + h).
    """
    # Calculate heuristic for the start node
    start_state.calculateHeuristic()

    priority_queue = []
    heapq.heappush(priority_queue, start_state)

    visited = set()
    visited.add(start_state)
    expanded_nodes = 0

    while priority_queue:
        # Pop the state with the lowest f-cost
        current_state = heapq.heappop(priority_queue)
        expanded_nodes += 1

        # In A*, goal check is typically done after popping from the queue
        if current_state.isGoal():
            return current_state, expanded_nodes

        for action in current_state.getPossibleActions():
            child_state = current_state.applyAction(action)

            if child_state not in visited:
                child_state.calculateHeuristic()
                visited.add(child_state)
                heapq.heappush(priority_queue, child_state)

    return None, expanded_nodes


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure correct number of arguments are provided
    if len(sys.argv) < 10:
        print("Usage: Tiles.py <9 numbers>")
        # Example: python Tiles.py 1 4 0 5 8 2 3 6 7
    else:
        # Parse input arguments into a numpy array
        try:
            input_list = [int(x) for x in sys.argv[1:10]]
            start_board = np.array(input_list, dtype=np.uint8).reshape(3, 3)
        except ValueError:
            print("Error: Inputs must be integers.")
            sys.exit(1)

        # --- Run BFS ---
        print("Algorithm: BFS")
        initial_state_bfs = PuzzleState(start_board)
        sol_bfs, exp_bfs = BFS(initial_state_bfs)

        if sol_bfs:
            path = get_solution_path(sol_bfs)
            print("Path:", " ".join(map(str, path)))
            print("Length:", len(path))
            print("Expanded:", exp_bfs)
        else:
            print("No solution found")

        print("-" * 30)

        # --- Run A* ---
        print("Algorithm: A*")
        initial_state_astar = PuzzleState(start_board)
        sol_astar, exp_astar = A_Star(initial_state_astar)

        if sol_astar:
            path = get_solution_path(sol_astar)
            print("Path:", " ".join(map(str, path)))
            print("Length:", len(path))
            print("Expanded:", exp_astar)
        else:
            print("No solution found")