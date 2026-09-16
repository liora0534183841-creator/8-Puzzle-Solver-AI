# 8-Puzzle AI Solver (BFS & A*)

An implementation of the classic 8-puzzle game solver in Python, utilizing un-informed (Breadth-First Search) and informed (A* Search) search algorithms to find the optimal path to the target state.

## 🛠️ Core Technologies
*   **Language:** Python
*   **Libraries:** `numpy` (for efficient memory and matrix operations), `heapq` (for priority queue implementation).
*   **Algorithms:** BFS (Breadth-First Search), A* (A-Star Search).

## 🧠 State Representation and Transitions
*   **State Space:** Each state is represented as a 3x3 `numpy` array of type `uint8` for optimal memory efficiency and fast indexing.
*   **Goal State:** The target configuration is `[[0, 1, 2], [3, 4, 5], [6, 7, 8]]`.
*   **Actions:** The empty tile (represented by `0`) can move 'Left', 'Right', 'Up', or 'Down'.
*   **Transition Model:** State transitions compute the new position using vector addition and generate a deep copy of the board to maintain state immutability.

## 📐 The Heuristic (A* Optimization)
The A* algorithm uses a highly optimized, admissible, and consistent heuristic to guide the search:
**h(n) = Manhattan Distance + (2 × Linear Conflict)**

*   **Manhattan Distance:** Calculates the absolute minimum number of moves each tile must make to reach its target position (sum of row and column distances).
*   **Linear Conflict:** Adds a penalty of 2 moves for every pair of tiles that are in their correct row or column but in reversed order, blocking each other. 
*   **Why it works:** This heuristic ensures that A* explores the most promising paths first without overestimating the cost, guaranteeing an optimal solution while drastically reducing the search space.

## 📊 Performance Comparison: BFS vs A*
Extensive testing demonstrates the profound efficiency of the A* algorithm over standard BFS:

**Test Case 1 (Standard Board - Path Length: 10):**
*   **BFS:** Expanded 357 nodes.
*   **A*:** Expanded only 11 nodes. *(~32x reduction in processing)*

**Test Case 2 (Edge Case - Path Length: 26):**
*   **BFS:** Expanded 151,283 nodes.
*   **A*:** Expanded only 1,490 nodes. *(~100x reduction in processing)*

**Conclusion:** While BFS suffers from exponential growth when the solution is distant, the custom heuristic guides A* efficiently, saving roughly 99% of processing overhead in complex scenarios.

## 🚀 How to Run
Run the script from the command line, providing the initial board configuration as 9 space-separated integers (0 represents the empty tile).

```bash
python Tiles.py 1 4 0 5 8 2 3 6 7
