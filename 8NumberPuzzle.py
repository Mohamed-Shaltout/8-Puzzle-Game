from collections import deque
import copy
import tkinter as tk
import math
import heapq
import time

level=0
goal_positions = {
    1: (0, 1), 2: (0, 2), 3: (1, 0),
    4: (1, 1), 5: (1, 2), 6: (2, 0),
    7: (2, 1), 8: (2, 2), 0: (0, 0)
}
# Define colors for GUI
color = "grey"
Goal = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
def print_gui(x):
        create_board_gui(x)
def print_path(path):
    if path:
        print("Path to solution:")
        for step, board_state in enumerate(path):
            print(f"Step {step}:")
            for row in board_state:
                print(row)
            print("\n")  # Separate each state for readabilit
        print("Path Cost is : "+str(step))
        print("Search Depth is : "+str(step))
    else:
        print("No solution found.")
def find_empty_tile(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j
    return None

def Generate_children(arr):
    children = []
    temp1 = copy.deepcopy(arr)
    temp2 = copy.deepcopy(arr)
    temp3 = copy.deepcopy(arr)
    temp4 = copy.deepcopy(arr)
    found = False

    for i, row in enumerate(arr):
        if 0 in row:
            j = row.index(0)
            found = True
            break

    if found: #1 left #2 right #3 up #4 down      3412

        if i == 1 and j == 1:
            children.extend([swap(temp1, 2, i, j), swap(temp2, 1, i, j), swap(temp3, 4, i, j), swap(temp4, 3, i, j)])
        elif i == 1 and j == 0:
            children.extend([swap(temp1, 2, i, j), swap(temp2, 4, i, j), swap(temp3, 3, i, j)])
        elif i == 2 and j == 0:
            children.extend([swap(temp1, 2, i, j), swap(temp2, 3, i, j)])
        elif i == 0 and j == 0:
            children.extend([swap(temp1, 2, i, j), swap(temp2, 4, i, j)])
        elif i == 0 and j == 1:
            children.extend([swap(temp1, 2, i, j), swap(temp2, 1, i, j), swap(temp3, 4, i, j)])
        elif i == 0 and j == 2:
            children.extend([swap(temp1, 1, i, j), swap(temp2, 4, i, j)])
        elif i == 1 and j == 2:
            children.extend([swap(temp1, 1, i, j), swap(temp2, 4, i, j), swap(temp3, 3, i, j)])
        elif i == 2 and j == 2:
            children.extend([swap(temp1, 1, i, j), swap(temp2, 3, i, j)])
        elif i == 2 and j == 1:
            children.extend([swap(temp1, 2, i, j), swap(temp2, 1, i, j), swap(temp3, 3, i, j)])
    return children

def A_star_Manhattan(initial_state):
    Frontier = []
    cost_so_far = {}
    initial_cost = manhattan_heuristic_2d(initial_state)
    heapq.heappush(Frontier, (initial_cost, initial_state, [], initial_cost))
    cost_so_far[tuple(map(tuple, initial_state))] = 0
    visited=0
    while Frontier:
        f, current_state, path, heuristic_cost = heapq.heappop(Frontier)

        if check_goal(current_state):
            return path + [(current_state, heuristic_cost)],visited

        for neighbor in Generate_children(current_state):
            new_cost = cost_so_far[tuple(map(tuple, current_state))] + 1
            neighbor_tuple = tuple(map(tuple, neighbor))

            if neighbor_tuple not in cost_so_far or new_cost < cost_so_far[neighbor_tuple]:
                visited+=1
                cost_so_far[neighbor_tuple] = new_cost
                heuristic = manhattan_heuristic_2d(neighbor)
                priority = new_cost + heuristic
                heapq.heappush(Frontier, (priority, neighbor, path + [(current_state, heuristic_cost)], heuristic))

    return None

def A_star_Euclidean(initial_state):
    Frontier = []
    cost_so_far = {}
    initial_cost = euclidean_heuristic_2d(initial_state)
    heapq.heappush(Frontier, (initial_cost, initial_state, [], initial_cost))
    cost_so_far[tuple(map(tuple, initial_state))] = 0
    visited=0

    while Frontier:
        f, current_state, path, heuristic_cost = heapq.heappop(Frontier)

        if check_goal(current_state):
            print("Path Cost: "+str(print_path(path)))
            return path + [(current_state, heuristic_cost)],visited

        for neighbor in Generate_children(current_state):
            new_cost = cost_so_far[tuple(map(tuple, current_state))] + 1
            neighbor_tuple = tuple(map(tuple, neighbor))

            if neighbor_tuple not in cost_so_far or new_cost < cost_so_far[neighbor_tuple]:
                visited+=1
                cost_so_far[neighbor_tuple] = new_cost
                heuristic = euclidean_heuristic_2d(neighbor)
                priority = new_cost + heuristic
                heapq.heappush(Frontier, (priority, neighbor, path + [(current_state, heuristic_cost)], heuristic))

    return None


def manhattan_heuristic_2d(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            if tile != 0:
                goal_row, goal_col = goal_positions[tile]
                distance += abs(goal_row - i) + abs(goal_col - j)
    return distance
def euclidean_heuristic_2d(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            if tile != 0:
                goal_row, goal_col = goal_positions[tile]
                distance += math.sqrt((goal_row - i) ** 2 + (goal_col - j) ** 2)
    return distance
def create_board_gui(solution_path, interval=1000):
    root = tk.Tk()
    root.title("8-Puzzle Solution Path")

    # Create a 3x3 grid of labels with larger cell sizes
    labels = [[tk.Label(root, text="", borderwidth=3, relief="solid", width=15, height=7,
                        font=("Times New Roman", 18)) for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            labels[i][j].grid(row=i, column=j)

    def update_board(step=0):
        if step < len(solution_path):
            board = solution_path[step]
            # Update each label with the current board state
            for i in range(3):
                for j in range(3):
                    value = board[i][j]
                    labels[i][j].config(text=str(value) if value != 0 else "",
                                        bg="white" if value != 0 else "lightgray")
            # Schedule the next update
            root.after(interval, update_board, step + 1)

    # Start the board updates
    update_board()

    root.mainloop()

def check_goal(arr):
    Goal = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    return arr == Goal


def swap(arr, case, index1, index2):
    if case == 1 and index2 > 0:  # left
        arr[index1][index2], arr[index1][index2 - 1] = arr[index1][index2 - 1], arr[index1][index2]
    elif case == 2 and index2 < len(arr[0]) - 1:  # right
        arr[index1][index2], arr[index1][index2 + 1] = arr[index1][index2 + 1], arr[index1][index2]
    elif case == 3 and index1 > 0:  # up
        arr[index1 - 1][index2], arr[index1][index2] = arr[index1][index2], arr[index1 - 1][index2]
    elif case == 4 and index1 < len(arr) - 1:  # down
        arr[index1 + 1][index2], arr[index1][index2] = arr[index1][index2], arr[index1 + 1][index2]
    return arr

class Board:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.children = []

def create_tree(value):
    return Board(value)

def traverse_tree(node, level=0):
    if node is not None:
        print("  " * level + "Node value (2D Array):")
        for row in node:
            print("  " * level + str(row))
        for child in node.children:
            traverse_tree(child, level + 1)

def bfsSearch(initial_board):
    root = Board(initial_board)
    queue = deque([root])
    visited = set()
    visited.add(tuple(map(tuple, initial_board)))
    depth=0
    while queue:
        current_node = queue.popleft()
        current_board = current_node.value

        if check_goal(current_board):
            path = []
            while current_node:
                path.append(current_node.value)
                current_node = current_node.parent
            print("The nodes expanded are : "+str(len(visited)))

            return path[::-1] 
        depth += 1
        for child_value in Generate_children(current_board):
            child_tuple = tuple(map(tuple, child_value))
            if child_tuple not in visited:
                visited.add(child_tuple)
                child_node = Board(child_value, current_node)
                queue.append(child_node )
                
    
    return None

def dfsSearch(currentState):
    root = Board(currentState)
    stack = [root]
    visited = set()
    visited.add(tuple(map(tuple, currentState)))

    while stack:
        currentNode = stack.pop()
        currentBoard = currentNode.value

        if check_goal(currentBoard):
            # Backtrack to build the path from root to goal
            path = []
            while currentNode:
                path.append(currentNode.value)
                currentNode = currentNode.parent
            print("The nodes expanded are : "+str(len(visited)))
            return path[::-1]  # Reverse the path to start from the initial state

        for child in Generate_children(currentBoard):
            childTuple = tuple(map(tuple, child))
            if childTuple not in visited:
                visited.add(childTuple)
                childNode = Board(child, currentNode)
                stack.append(childNode)

    return None


def HelperdfsSearch(currentState, max_depth): 
    root = Board(currentState)
    stack = [(root, 0)]  
    visited = {tuple(map(tuple, currentState))}

    while stack:
        currentNode, currentDepth = stack.pop()
        currentBoard = currentNode.value

        if check_goal(currentBoard):

            return currentNode,len(visited)  

        if currentDepth < max_depth: 
            currentDepth += 1
            for child in Generate_children(currentBoard):
                childTuple = tuple(map(tuple, child))
                if childTuple not in visited:
                    visited.add(childTuple)
                    childNode = Board(child, currentNode)
                    stack.append((childNode, currentDepth)) 

    return None,len(visited) 

def idsSearch(currentState):
    i = 0
    while True:
        currentNode,len = HelperdfsSearch(currentState, i)
        if currentNode and check_goal(currentNode.value):
            print("The nodes expanded are : "+str(len))
            return currentNode
        i += 1

def is_within_range(arr):
    for i in range(0, 3):
        if (arr[i].isdigit()):
            if (int(arr[i]) > 8 or int(arr[i]) < 0):
                return False
        else:
            return False
    return True


def check_duplicates(arr):
    seen = set()
    for item in arr:
        if item in seen:
            return False
        seen.add(item)

    return True


def verify(row_values, distinct_numbers):
    if (check_duplicates(row_values)):
        for value in row_values:
            if value in distinct_numbers:
                return True
        if (is_within_range(row_values)):
            for value in row_values:
                distinct_numbers.append(value)
            return False
        else:
            print("Enter Numbers Within 0 --> 8")
            return True
    else:
        return True


def get_path(goal_node):
    path = []
    current = goal_node
    while current is not None:
        path.append(current.value)
        current = current.parent
    return path[::-1]


def count_inversions_2d(board):
    flattened_board = [tile for row in board for tile in row if tile != 0]
    inversions = 0
    for i in range(len(flattened_board)):
        for j in range(i + 1, len(flattened_board)):
            if flattened_board[i] > flattened_board[j]:
                inversions += 1
    return inversions




#print(manhattan_heuristic_2d(My_board))
#print(euclidean_heuristic_2d(My_board,Goal))

#print(root.Manhattan_distance)
#solution_path_Manhattan = A_star_Manhattan(My_board)

#A_Star_Search_EuclideanDistance(My_board)
#goal_node = bfsSearch(My_board)
#solution_path = get_path(goal_node)
#for step in solution_path:
 #   create_board_gui(step)
def menu():
    print("Enter the elements of the 2D array row by row (space-separated):")
    My_board = []
    distinct_numbers = []

    for i in range(3):
        while True:
            row_input = input(f"Row {i + 1}: ")
            row_values = row_input.split()

            if len(row_values) != 3 or verify(row_values, distinct_numbers):
                print(f"Please enter exactly 3 distinct elements.")
                print(distinct_numbers)
            elif not all(value.isdigit() for value in row_values):
                print(f"Please enter exactly 3 elements of type integer.")
            else:
                My_board.append([int(value) for value in row_values])
                break

    num_inversions = count_inversions_2d(My_board)
    root = Board(My_board)
    if num_inversions % 2 != 0:
        print("Number of inversions is odd, so the puzzle is unsolvable.")
        exit(-1)
    else:
        print("Number of inversions is even, so the puzzle is solvable.")


    print("Select the search algorithm you want to use:")
    print("1. Breadth-First Search (BFS)")
    print("2. Depth-First Search (DFS)")
    print("3. Iterative-Depth Search (IDS)")
    print("4. A* Search Manhattan")
    print("5. A* Search Eculdien")

    choice = input("Enter the number of your choice: ")
    x=0
    y=0

    # Execute the chosen search algorithm and measure time
    

    if choice == "1":
        print("\nStarting Breadth-First Search (BFS)...")
        start_time = time.time()  # Start time
        z =bfsSearch(My_board)
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Running time: {elapsed_time:.4f} seconds")
        print_path(z)
        print_gui(z)
    elif choice == "2":
        print("\nStarting Depth-First Search (DFS)...")
        start_time = time.time()  # Start time
        l=dfsSearch(My_board)
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Running time: {elapsed_time:.4f} seconds")
        print_path(l)
        print_gui(l)
    elif choice == "3":
        print("\nStarting IDS Search...")
        start_time = time.time()  # Start time
        current_board=idsSearch(My_board)
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Running time: {elapsed_time:.4f} seconds")
        solution_path = get_path(current_board)
        print_path(solution_path)
        print_gui(solution_path)
    elif choice == "4":
        print("\nStarting A* Search... with Manhattan Heuristic")
        start_time = time.time()  # Start time
        x,visited=A_star_Manhattan(My_board)
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Running time: {elapsed_time:.4f} seconds")
        print("The Nodes Expanded are "+str(visited))
    elif choice == "5":
        print("\nStarting A* Search... with Euclidean Heuristic")
        start_time = time.time()  # Start time
        y,visited=A_star_Euclidean(My_board)
        end_time = time.time()  # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Running time: {elapsed_time:.4f} seconds")
        print("The Nodes Expanded are "+str(visited))
    else:
        print("Invalid choice. Please enter 1, 2,3,4, or 5.")
        return
    list = []
    if x:
        print("Solution path with Manhattan distances:")
        for step, (state, distance) in enumerate(x):
            print(f"Step {step} (Manhattan distance: {distance}):")
            list.append(state)
            for row in state:
                print(row)
            print()
        print_gui(list)
        print("path cost:"+str(step))

        #solution_path_Euclidean = A_star_Euclidean(My_board)
    if y:
        print("Solution path with Euclidean distances:")
        for step, (state, distance) in enumerate(y):
            print(f"Step {step} (Euclidean distance: {distance}):")
            list.append(state)
            for row in state:
                print(row)
            print()
        print_gui(list)
        print("path cost:" +str(step))
    end_time = time.time()  # End time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Running time: {elapsed_time:.4f} seconds")

    # Run the menu


if __name__ == "__main__":
    menu()