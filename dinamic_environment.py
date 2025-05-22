import numpy as np
import random
import heapq
import time
import matplotlib.pyplot as plt

class Grid:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.grid = np.zeros((width, height))
        self.update_obstacles(obstacles)

    def update_obstacles(self, obstacles):
        self.grid.fill(0)
        self.obstacles = obstacles
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
    
CLUSTERS = True

def dijkstra(grid, start, goal):
    queue = [(0, start)]
    distances = {start: 0}
    came_from = {start: None}
    current = start

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
    current = start

    while queue:

        current = queue.popleft()
        for neighbor in grid.neighbors(*current):

            if neighbor not in wave:
                queue.append(neighbor)
                wave[neighbor] = wave[current] + 1

    current = start
    path = [current]
    if goal not in wave or current not in wave:
        #print("Ошибка: цель недостижима")
        return []

    while (current != goal):
        neighbors = grid.neighbors(*current)
        #print(f"Current position: {current}, Neighbors: {neighbors}")
        current = min(neighbors, key=lambda x: wave.get(x, float('inf')))
        path.append(current)

    return path

def potential_field(grid, start, goal, alpha=1.0, beta=5.0, gamma=0.1, max_iter=1000):

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
    
    collisions = 0
    visited = set()
    current = start
    previous_step = start
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
                next_step = random.choice(grid.neighbors(*current))

        if grid.is_obstacle(*next_step):
            collisions += 1

        if current != next_step:
            current = next_step
            path.append(current)
            visited.add(current)

        if CLUSTERS:
            dynamic_obstacles_step(grid, goal, current)
        else:
            dynamic_obstacles_random_step(grid, goal, current)

    return path, collisions

def generate_obstacle_clusters(grid, goal, current, cluster_size=30, spread=3):
    num_clusters = round(grid.width * grid.height * 0.01)
    while True:
        obstacles = []
        locates = []
        for _ in range(num_clusters):
            # Выбираем центр кластера и его направление
            direction = random.randint(0, 3)
            cx = random.randint(3, grid.width - 4)
            cy = random.randint(3, grid.height - 4)
            for _ in range(cluster_size):
                # Распределяем препятствия около центра с заданным разбросом
                dx = int(random.gauss(0, spread))
                dy = int(random.gauss(0, spread))
                x, y = cx + dx, cy + dy
                if 0 <= x < grid.width and 0 <= y < grid.height:
                    obstacles.append((x, y, direction))
                    locates.append((x, y))
        if goal not in locates and current not in locates:
            grid.update_obstacles(obstacles)
            break

def generate_obstacles_points(grid, goal, current):
    while True:
        locates = []
        new_obstacles = [(np.random.randint(0, grid.width), np.random.randint(0, grid.height), np.random.randint(0, 3)) for _ in range(NUMBER_OF_OBSTACLES)]
        for new_obstacle in new_obstacles:
            locates.append(new_obstacle[:2])
        if goal not in locates and current not in locates:
            grid.update_obstacles(new_obstacles)
            break

def dynamic_obstacles_step(grid, goal, current):
    dir_map = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    new_obstacles = []
    locates = []

    for obs in grid.obstacles:
        x, y, d = obs
        dx, dy = dir_map[d]
        nx, ny = x + dx, y + dy
        if not grid.in_bounds(nx, ny):
            nx, ny = obstacle_return(nx, ny)
        new_obstacles.append((nx, ny, d))
        locates.append((nx, ny))

    if goal not in locates and current not in locates:
        grid.update_obstacles(new_obstacles)

def dynamic_obstacles_random_step(grid, goal, current):
    directions = ((0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0))
    while True:
        new_obstacles = []
        locates = []
        for obstacle in grid.obstacles:
            is_mooving = random.randint(0, 4)
            keep_direction = random.randint(0, 1)
            change_direction = random.randint(0, 3)
            if is_mooving > 0:
                if keep_direction == 0:
                    obstacle = (obstacle[0], obstacle[1], change_direction)
                new_obstacle = (obstacle[0] + directions[obstacle[2]][0], 
                                obstacle[1] + directions[obstacle[2]][1], 
                                obstacle[2])
                if grid.in_bounds(new_obstacle[0], new_obstacle[1]):
                    new_obstacles.append(new_obstacle)
                else:
                    x, y = obstacle_return(new_obstacle[0], new_obstacle[1])
                    new_obstacle = (x, y, obstacle[2])
                    new_obstacles.append(new_obstacle)
            else:
                new_obstacles.append(obstacle)
        for new_obstacle in new_obstacles:
            locates.append(new_obstacle[:2])

        if (goal not in locates) and (current not in locates):
            grid.update_obstacles(new_obstacles)
            #print("Dynamic obstacles moved:", new_obstacles)
            break

def obstacle_return(x, y):
    nx, ny = x, y
    if x < 0:
        nx = x + grid.width
    if x > grid.width - 1:
        nx = x - grid.width
    if y < 0:
        ny = y + grid.height
    if y > grid.height - 1:
        ny = y - grid.height
    return nx, ny

WIDTH = 50
HEIGHT = 50
NUMBER_OF_OBSTACLES = round(WIDTH * HEIGHT * 0.3)

start = (2, 2)
goal = (WIDTH - 3, HEIGHT - 3)

grid = Grid(WIDTH, HEIGHT, [])

paths = {}
timings = {}

for name, func in {
    "Dijkstra": dijkstra,
    "A*": a_star,
    "D*": d_star,
    "Wavefront": wavefront,
}.items():
    start_time = time.time()
    current = start
    path = [current]
    if CLUSTERS:
        generate_obstacle_clusters(grid, goal, current)
    else:
        generate_obstacles_points(grid, goal, current)
    while current != goal:
        path_func = func(grid, current, goal)
        if len(path_func) > 1:
            next_step = path_func[1]
            current = next_step
            path.append(current)
        dynamic_obstacles_random_step(grid, goal, current)
    elapsed = round((time.time() - start_time) * 1000, 3)
    paths[name] = path
    timings[name] = elapsed
    print(f"{name} Time: {elapsed} ms, Path length: {len(path)}")

start_time = time.time()
path_potential_field, collisions = potential_field(grid, start, goal)
potential_field_time = time.time() - start_time

print(f"Potential Field Time: {round(potential_field_time * 1000, 3)} ms, Path length: {len(path_potential_field)}")

fig, ax = plt.subplots(figsize=(10, 10))
grid_matrix = np.zeros((WIDTH, HEIGHT))
#for x, y, _ in grid.obstacles:
#    grid_matrix[x, y] = 1
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
xs, ys = zip(*path_potential_field)
ax.plot(xs, ys, label=f"Potential Field Time ({round(potential_field_time * 1000, 3)} ms)", color=colors["Potential Field"], linewidth=1)

ax.plot(*start, 'go')
ax.plot(*goal, 'ro')
ax.legend()
plt.title("Алгоритмы в динамической среде")
plt.grid(False)
plt.show()
