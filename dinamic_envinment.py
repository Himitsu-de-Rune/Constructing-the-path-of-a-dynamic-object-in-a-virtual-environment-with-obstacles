import numpy as np
import random
import heapq
import time

class Grid:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.update_obstacles(obstacles)

    def update_obstacles(self, obstacles):
        self.grid.fill(0)
        for obstacle in obstacles:
            self.grid[obstacle[0], obstacle[1]] = 1

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

def dijkstra(grid, start, goal, update_interval=0.00002):
    queue = [(0, start)]
    distances = {start: 0}
    came_from = {start: None}
    start_time = time.time()
    current = start

    while queue:
        current_distance, current = heapq.heappop(queue)

        if time.time() - start_time > update_interval:
            start_time = time.time()
            dynamic_obstacles_update(grid, start, goal, current)

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

def a_star(grid, start, goal, update_interval=0.00002):
    queue = [(0, start)]
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    came_from = {start: None}
    start_time = time.time()
    current = start

    while queue:
        _, current = heapq.heappop(queue)

        if time.time() - start_time > update_interval:
            start_time = time.time()
            dynamic_obstacles_update(grid, start, goal, current)

        if current == goal:
            break

        for neighbor in grid.neighbors(*current):
            new_g_cost = g_costs[current] + 1

            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor, goal)
                heapq.heappush(queue, (f_cost, neighbor))
                came_from[neighbor] = current

    path = []
    if goal in came_from:
        current = goal
        while current:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
    else:
        print("Цель не была достигнута")
        return []
    return path

def d_star(grid, start, goal, update_interval=0.00002):
    queue = [(0, start)]
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    came_from = {start: None}
    visited = set()
    start_time = time.time()
    current = start

    while queue:
        _, current = heapq.heappop(queue)

        if time.time() - start_time > update_interval:
            start_time = time.time()
            dynamic_obstacles_update(grid, start, goal, current)

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
                heapq.heappush(queue, (f_cost, neighbor))
                came_from[neighbor] = current

    path = []
    if goal in came_from:
        current = goal
        while current:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
    else:
        print("Цель не была достигнута")
        return []
    return path

def wavefront(grid, start, goal, update_interval=0.00002):
    from collections import deque
    queue = deque([goal])
    wave = {goal: 0}
    start_time = time.time()
    current = start

    while queue:
        current = queue.popleft()
        for neighbor in grid.neighbors(*current):

            if neighbor not in wave:
                queue.append(neighbor)
                wave[neighbor] = wave[current] + 1

        if time.time() - start_time > update_interval:
            start_time = time.time()
            dynamic_obstacles_update(grid, start, goal, current)

    current = start
    last_current = ''
    path = [current]

    while (current != goal):
        neighbors = grid.neighbors(*current)
        last_current = current
        #print(f"Current position: {current}, Neighbors: {neighbors}")
        if neighbors != []:
            current = min(neighbors, key=lambda x: wave.get(x, float('inf')))
        if last_current != current:
            path.append(current)

        if time.time() - start_time > update_interval:
            start_time = time.time()
            dynamic_obstacles_update(grid, start, goal, current)

    return path

def potential_field(grid, start, goal, alpha=1.0, beta=5.0, gamma=0.1, max_iter=500, update_interval=0.00002):

    def attractive_potential(x, y):
        return alpha * ((x - goal[0]) ** 2 + (y - goal[1]) ** 2)
    
    def repulsive_potential(x, y):
        min_dist = float('inf')
        for i in range(grid.width):
            for j in range(grid.height):
                if grid.is_obstacle(i, j):
                    dist = np.sqrt((x - i) ** 2 + (y - j) ** 2)
                    if dist < min_dist:
                        min_dist = dist
        if min_dist == 0:
            return float('inf')
        return beta / min_dist
    
    visited = set()
    current = start
    path = [current]
    visited.add(current)
    start_time = time.time()

    for _ in range(max_iter):
        if current == goal:
            break

        potentials = []
        for neighbor in grid.neighbors(*current):
            x, y = neighbor
            potential = (attractive_potential(x, y) + 
                         repulsive_potential(x, y) +
                         gamma * heuristic(neighbor, goal))
            potentials.append((potential, neighbor))
        
        if potentials:
            _, next_step = min(potentials, key=lambda x: x[0])
        
            if next_step in visited:
                potentials.remove((_, next_step))
                if potentials:
                    _, next_step = min(potentials, key=lambda x: x[0])

        if current != next_step:
            current = next_step
            path.append(current)
            visited.add(current)

        if time.time() - start_time > update_interval:
            start_time = time.time()
            dynamic_obstacles_update(grid, start, goal, current)

    return path

def dynamic_obstacles_update(grid, start, goal, current):
    while True:
        new_obstacles = [(np.random.randint(0, grid.width), np.random.randint(0, grid.height)) for _ in range(30)]
        if start not in new_obstacles and goal not in new_obstacles and current not in new_obstacles:
            grid.update_obstacles(new_obstacles)
            #print("Dynamic obstacles updated:", new_obstacles)
            break

initial_obstacles = [(4, 0), (4, 1), (3, 1), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (4, 5), (3, 6), (5, 8), (6, 7), (6, 6), (7, 6), (8, 6), (8, 7), (8, 8)]

start = (0, 0)
goal = (7, 7)

grid = Grid(10, 10, initial_obstacles)
start_time = time.time()
path_dijkstra = dijkstra(grid, start, goal)
dijkstra_time = time.time() - start_time

#print("Dijkstra Path:", path_dijkstra)
print(f"Dijkstra Time: {round(dijkstra_time * 1000, 3)} ms, Dijkstra PathLength: {len(path_dijkstra)}")

grid = Grid(10, 10, initial_obstacles)
start_time = time.time()
path_a_star = a_star(grid, start, goal)
a_star_time = time.time() - start_time

#print("A* Path:", path_a_star)
print(f"A* Time: {round(a_star_time * 1000, 3)} ms, A* PathLength: {len(path_a_star)}")

grid = Grid(10, 10, initial_obstacles)
start_time = time.time()
path_d_star = d_star(grid, start, goal)
d_star_time = time.time() - start_time

#print("D* Path:", path_d_star)
print(f"D* Time: {round(d_star_time * 1000, 3)} ms, D* PathLength: {len(path_d_star)}")

grid = Grid(10, 10, initial_obstacles)
start_time = time.time()
path_wavefront = wavefront(grid, start, goal)
wavefront_time = time.time() - start_time

#print("Wavefront Path:", path_wavefront)
print(f"Wavefront Time {round(wavefront_time * 1000, 3)} ms, Wavefront PathLength: {len(path_wavefront)}")

grid = Grid(10, 10, initial_obstacles)
start_time = time.time()
path_potential_field = potential_field(grid, start, goal)
potential_field_time = time.time() - start_time

#print("Potential Field Path:", path_potential_field)
print(f"Potential Field Time: {round(potential_field_time * 1000, 3)} ms, Potential Field PathLength: {len(path_potential_field)}")
