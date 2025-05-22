import numpy as np
import random
import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def dynamic_obstacles_generate(grid, goal, current):
    while True:
        locates = []
        new_obstacles = [(np.random.randint(0, grid.width), np.random.randint(0, grid.height), np.random.randint(0, 3)) for _ in range(NUMBER_OF_OBSTACLES)]
        for new_obstacle in new_obstacles:
            locates.append(new_obstacle[:2])
        if (goal not in locates) and (current not in locates):
            grid.update_obstacles(new_obstacles)
            break

def dynamic_obstacles_step(grid, goal, current):
    directions = ((0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0))
    new_obstacles = []
    locates = []
    for obstacle in grid.obstacles:
        is_mooving = random.randint(0, 5)
        is_keeping_direction = random.randint(0, 1)
        new_direction = random.randint(0, 3)
        if is_mooving > 0:
            if is_keeping_direction == 0:
                obstacle = (obstacle[0], obstacle[1], new_direction)
            new_obstacle = (obstacle[0] + directions[obstacle[2]][0],
                            obstacle[1] + directions[obstacle[2]][1],
                            obstacle[2])
            if grid.in_bounds(new_obstacle[0], new_obstacle[1]):
                new_obstacles.append(new_obstacle)
            else:
                x, y = obstacle_return(new_obstacle[0], new_obstacle[1], grid)
                new_obstacle = (x, y, obstacle[2])
                new_obstacles.append(new_obstacle)
        else:
            new_obstacles.append(obstacle)
    for new_obstacle in new_obstacles:
        locates.append(new_obstacle[:2])
    if (goal not in locates) and (current not in locates):
        grid.update_obstacles(new_obstacles)

def obstacle_return(x, y, grid):
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

class HybridAgent:
    def __init__(self, grid, start, goal, sensor_range = 4):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.q_table = {}
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.sensor_range = sensor_range
        self.known_map = set()
        self.training_log = []

    def state(self, pos):
        x, y = pos
        state = []
        for dx, dy in self.actions:
            nx, ny = x + dx, y + dy
            if self.grid.in_bounds(nx, ny):
                state.append(int(self.grid.is_obstacle(nx, ny)))
            else:
                state.append(1)
        return tuple(state)

    def choose_action(self, s):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.q_table.get((s, a), 0) for a in self.actions]
        max_q = max(q_values)
        return self.actions[q_values.index(max_q)]

    def learn(self, episodes=500, max_steps_per_episode=5000):
        for ep in range(episodes):
            current = self.start
            dynamic_obstacles_step(self.grid, self.goal, current)
            steps = 0
            visited = set()
            visited.add(current)

            while current != self.goal and steps < max_steps_per_episode:
                steps += 1
                s = self.state(current)
                a = self.choose_action(s)
                nx, ny = current[0] + a[0], current[1] + a[1]
                dist_old = heuristic(current, self.goal)
                next_s = s

                if not self.grid.in_bounds(nx, ny) or self.grid.is_obstacle(nx, ny):
                    reward = -10
                else:
                    dist_new = heuristic((nx, ny), self.goal)
                    reward = -1 if (nx, ny) != self.goal else 100
                    reward += (dist_old - dist_new) * 0.5
                    if (nx, ny) in visited:
                        reward -= 2
                    next_s = self.state((nx, ny))
                    current = (nx, ny)
                    visited.add(current)

                old_q = self.q_table.get((s, a), 0)
                next_q = max([self.q_table.get((next_s, next_a), 0) for next_a in self.actions])
                self.q_table[(s, a)] = old_q + self.alpha * (reward + self.gamma * next_q - old_q)

            #print(f"Episode {ep}, Reached: {current == self.goal}, Steps: {steps}")

    def update_known_map(self, pos):
        x, y = pos
        self.known_map.clear
        for dx in range(-self.sensor_range, self.sensor_range + 1):
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                nx, ny = x + dx, y + dy
                if self.grid.in_bounds(nx, ny):
                    self.known_map.add((nx, ny))

    def filtered_neighbors(self, x, y):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        result = []
        for direction in directions:
            nx, ny = x + direction[0], y + direction[1]
            if self.grid.in_bounds(nx, ny) and (not self.grid.is_obstacle(nx, ny) or (nx, ny) not in self.known_map):
                result.append((nx, ny))
        return result
    
    def a_star(self, start, goal):
        queue = [(0, start)]
        g_costs = {start: 0}
        f_costs = {start: heuristic(start, goal)}
        came_from = {start: None}
        current = start

        while queue:
            _, current = heapq.heappop(queue)

            if current == goal:
                break

            for neighbor in self.filtered_neighbors(*current):
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
            return []
        return path

    def run(self, max_steps=None):
        if max_steps is None:
            max_steps = self.grid.width * self.grid.height * 5

        current = self.start
        self.update_known_map(current)
        full_path = self.a_star(self.start, self.goal)
        if not full_path:
            print("Маршрут не найден")
            return [], 0

        path_taken = [current]
        step_index = 1  # текущий индекс в маршруте A*
        collisions = 0
        steps = 0

        while current != self.goal and steps < max_steps:
            steps += 1
            print(f"Step {steps}: Agent at {current}")

            dynamic_obstacles_step(self.grid, self.goal, current)

            # Если ещё на маршруте и не достигли конца A*
            if step_index < len(full_path):
                next_step = full_path[step_index]

                if self.grid.is_obstacle(*next_step):
                    print(f"Obstacle detected at {next_step}. Executing local move.")

                    # Выполняем локальное движение
                    s = self.state(current)
                    a = self.choose_action(s)
                    nx, ny = current[0] + a[0], current[1] + a[1]

                    if self.grid.in_bounds(nx, ny) and not self.grid.is_obstacle(nx, ny):
                        current = (nx, ny)
                        self.update_known_map(current)
                        path_taken.append(current)
                    else:
                        print("Local move blocked. Staying in place.")
                        collisions += 1

                    full_path = self.a_star(current, self.goal)
                    if full_path:
                        step_index = 1
                else:
                    # Шагаем по маршруту A*
                    current = next_step
                    self.update_known_map(current)
                    path_taken.append(current)
                    step_index += 1
            else:
                print(f"Path can't be laid from {current}. Executing local move.")

                # Выполняем локальное движение
                s = self.state(current)
                a = self.choose_action(s)
                nx, ny = current[0] + a[0], current[1] + a[1]

                if self.grid.in_bounds(nx, ny) and not self.grid.is_obstacle(nx, ny):
                    current = (nx, ny)
                    self.update_known_map(current)
                    path_taken.append(current)
                else:
                    print("Local move blocked. Staying in place.")
                    collisions += 1

                full_path = self.a_star(current, self.goal)
                if full_path:
                    step_index = 1

        return path_taken, collisions


def visualize_path(grid, path, title="Path Visualization"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, grid.width + 1, 1))
    ax.set_yticks(np.arange(0, grid.height + 1, 1))
    ax.grid(True)

    for obs in grid.obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

    for i in range(len(path) - 1):
        x0, y0 = path[i]
        x1, y1 = path[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        ax.arrow(x0 + 0.5, y0 + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

    start_patch = patches.Rectangle((path[0][0], path[0][1]), 1, 1, facecolor='green')
    goal_patch = patches.Rectangle((path[-1][0], path[-1][1]), 1, 1, facecolor='red')
    ax.add_patch(start_patch)
    ax.add_patch(goal_patch)

    plt.gca().invert_yaxis()
    plt.show()

WIDTH = 50
HEIGHT = 50
NUMBER_OF_OBSTACLES = round(WIDTH * HEIGHT * 0.3)
grid = Grid(WIDTH, HEIGHT, [])
start = (2, 2)
goal = (WIDTH - 3, HEIGHT - 3)
dynamic_obstacles_generate(grid, goal, start)
print("Running hybrid agent...")
agent = HybridAgent(grid, start, goal)
agent.learn()
start_time = time.time()
path, collisions = agent.run()
elapsed_time = time.time() - start_time
print(f"Hybrid Agent Time: {round(elapsed_time * 1000, 3)} ms, PathLength: {len(path)}, Collisions: {collisions}")
visualize_path(grid, path, title="Hybrid Agent Path")