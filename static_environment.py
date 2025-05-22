import numpy as np
import random
import heapq
import time
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
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
    current = start

    while queue:
        _, current = heapq.heappop(queue)

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
        #print("Цель не была достигнута")
        return []
    return path

def d_star(grid, start, goal):
    queue = [(0, start)]
    g_costs = {start: 0}
    f_costs = {start: heuristic(start, goal)}
    came_from = {start: None}
    visited = set()
    current = start

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
        #print("Цель не была достигнута")
        return []
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

def potential_field(grid, start, goal, alpha=1.0, beta=5.0, gamma=0.1, max_iter=500):

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
        
        _, next_step = min(potentials, key=lambda x: x[0])
        
        if next_step in visited:
            next_step = random.choice(grid.neighbors(*current))

        current = next_step
        path.append(current)
        visited.add(current)

    return path

def generate_orthogonal_maze(width, height, start, goal):
    maze = np.ones((width, height), dtype=int)
    # Стартовая точка и шаги по нечетным координатам
    for x in range(1, width, 2):
        for y in range(1, height, 2):
            maze[x, y] = 0
            directions = []
            if x > 2: directions.append((-2, 0))
            if y > 2: directions.append((0, -2))
            if directions:
                dx, dy = random.choice(directions)
                maze[x + dx // 2, y + dy // 2] = 0

    obstacles = [(x, y) for x in range(width) for y in range(height) if maze[x, y] == 1 and (x, y) != start and (x, y) !=  goal]
    return obstacles

def generate_gradient_obstacles(width, height, start, goal, max_density=0.4):
    obstacles = []
    for x in range(width):
        # Плотность препятствий увеличивается от 0 до max_density вдоль оси X
        col_density = (x / width) * max_density
        for y in range(height):
            if random.random() < col_density:
                if (x, y) != start and (x, y) !=  goal:
                    obstacles.append((x, y))
    return obstacles

def generate_obstacle_clusters(width, height, num_clusters=30, cluster_size=30, spread=2):
    obstacles = set()
    for _ in range(num_clusters):
        # Выбираем центр кластера
        cx = random.randint(3, width - 4)
        cy = random.randint(3, height - 4)
        for _ in range(cluster_size):
            # Распределяем препятствия около центра с заданным разбросом
            dx = int(random.gauss(0, spread))
            dy = int(random.gauss(0, spread))
            x, y = cx + dx, cy + dy
            if 0 <= x < width and 0 <= y < height:
                obstacles.add((x, y))
    return list(obstacles)

WIDTH = 50
HEIGHT = 50

start = (2, 2)
goal = (WIDTH - 3, HEIGHT - 3)

obstacles = generate_orthogonal_maze(WIDTH, HEIGHT, start, goal)
#obstacles = generate_gradient_obstacles(WIDTH, HEIGHT, start, goal)
#obstacles = generate_obstacle_clusters(WIDTH, HEIGHT)
grid = Grid(WIDTH, HEIGHT, obstacles)

paths = {}
timings = {}

for name, func in {
    "Dijkstra": dijkstra,
    "A*": a_star,
    "D*": d_star,
    "Wavefront": wavefront,
    "Potential Field": potential_field
}.items():
    start_time = time.time()
    path = func(grid, start, goal)
    elapsed = round((time.time() - start_time) * 1000, 3)
    paths[name] = path
    timings[name] = elapsed
    print(f"{name} Time: {elapsed} ms, Path length: {len(path)}")
    print(path)

fig, ax = plt.subplots(figsize=(10, 10))
grid_matrix = np.zeros((WIDTH, HEIGHT))
for x, y in obstacles:
    grid_matrix[x, y] = 1
ax.imshow(grid_matrix.T, cmap='Greys', origin='lower')

colors = {
    "Dijkstra": 'blue',
    "A*": 'green',
    "D*": 'orange',
    "Wavefront": 'purple',
    "Potential Field": 'red'
}
for name, path in paths.items():
    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, label=f"{name} ({timings[name]} ms)", color=colors[name], linewidth=1)

ax.plot(*start, 'go')
ax.plot(*goal, 'ro')
ax.legend()
plt.title("Алгоритмы в лабиринтовой среде")
plt.grid(False)
plt.show()
