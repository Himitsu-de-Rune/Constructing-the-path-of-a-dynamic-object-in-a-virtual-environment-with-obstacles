import numpy as np
import heapq
import time

class Grid:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        for obstacle in obstacles:
            self.grid[obstacle[0], obstacle[1]] = 1  # 1 означает препятствие

    def is_obstacle(self, x, y):
        return self.grid[x, y] == 1

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x, y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        result = []
        for direction in directions:
            nx, ny = x + direction[0], y + direction[1]
            if self.in_bounds(nx, ny) and not self.is_obstacle(nx, ny):
                result.append((nx, ny))
        return result

def dijkstra(grid, start, goal):
    queue = [(0, start)]
    distances = {start: 0}
    came_from = {start: None}

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current == goal:
            break

        for neighbor in grid.neighbors(*current):
            new_distance = current_distance + 1
            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))
                came_from[neighbor] = current

    path = []
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    queue = [(0, start)]
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    came_from = {start: None}

    while queue:
        _, current = heapq.heappop(queue)

        if current == goal:
            break

        for neighbor in grid.neighbors(*current):
            new_g_cost = g_costs[current] + 1
            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor, goal)
                f_costs[neighbor] = f_cost
                heapq.heappush(queue, (f_cost, neighbor))
                came_from[neighbor] = current

    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def d_star(grid, start, goal):
    queue = [(0, start)]
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    came_from = {start: None}
    visited = set()

    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        visited.add(current)

        for neighbor in grid.neighbors(*current):
            if neighbor in visited:
                continue
            new_g_cost = g_costs[current] + 1
            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor, goal)
                f_costs[neighbor] = f_cost
                heapq.heappush(queue, (f_cost, neighbor))
                came_from[neighbor] = current

    path = []
    current = goal
    while current:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def wavefront(grid, start, goal):
    from collections import deque
    queue = deque([goal])
    wave = {goal: 0}

    while queue:
        current = queue.popleft()
        for neighbor in grid.neighbors(*current):
            if neighbor not in wave:
                queue.append(neighbor)
                wave[neighbor] = wave[current] + 1

    current = start
    path = [current]

    while current != goal:
        neighbors = grid.neighbors(*current)
        current = min(neighbors, key=lambda x: wave.get(x, float('inf')))
        path.append(current)

    return path

# Создание сетки 10x10 с препятствиями
obstacles = [(4, 0), (4, 1), (3, 1), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 5), (3, 6), (5, 8), (6, 7), (6, 6), (7, 6), (8, 6), (8, 7), (8, 8)]
grid = Grid(10, 10, obstacles)

start = (0, 0)
goal = (7, 7)

# Измерение времени для алгоритма Дейкстры
start_time = time.time()
path_dijkstra = dijkstra(grid, start, goal)
dijkstra_time = time.time() - start_time

# Измерение времени для алгоритма A*
start_time = time.time()
path_a_star = a_star(grid, start, goal)
a_star_time = time.time() - start_time

# Измерение времени для алгоритма D*
start_time = time.time()
path_d_star = d_star(grid, start, goal)
d_star_time = time.time() - start_time

# Измерение времени для волнового алгоритма
start_time = time.time()
path_wavefront = wavefront(grid, start, goal)
wavefront_time = time.time() - start_time

print("Dijkstra Path:", path_dijkstra)
print(f"Dijkstra Time: {round(dijkstra_time * 1000, 3)} ms")

print("A* Path:", path_a_star)
print(f"A* Time: {round(a_star_time * 1000, 3)} ms")

print("D* Path:", path_d_star)
print(f"D* Time: {round(d_star_time * 1000, 3)} ms")

print("Wavefront Path:", path_wavefront)
print(f"Wavefront Time {round(wavefront_time * 1000, 3)} ms")