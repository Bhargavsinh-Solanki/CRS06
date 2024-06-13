import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
GRID_SIZE = 30
NUM_OBJECTS = 300
NUM_ROBOTS = 200
SIMULATION_STEPS = 5000
PICK_SCALE = 1.5
DROP_SCALE = 2.0

# Define robot class
class Robot:
    def __init__(self, is_anti_agent=False):
        self.is_anti_agent = is_anti_agent
        self.x, self.y = np.random.randint(0, GRID_SIZE, 2)
        self.carrying = False

    def move(self):
        self.x = (self.x + np.random.choice([-1, 0, 1])) % GRID_SIZE
        self.y = (self.y + np.random.choice([-1, 0, 1])) % GRID_SIZE

    def neighborhood_density(self, grid):
        total_objects = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                nx, ny = (self.x + i) % GRID_SIZE, (self.y + j) % GRID_SIZE
                total_objects += grid[nx, ny]
        return total_objects

    def pick_up(self, grid):
        if self.carrying or grid[self.x, self.y] <= 0:
            return
        density = self.neighbourhood_density(grid)
        pick_prob = 1 - (1 / (1 + PICK_SCALE * density)) if self.is_anti_agent else (1 / (1 + PICK_SCALE * density))
        if random.random() < pick_prob:
            grid[self.x, self.y] -= 1
            self.carrying = True

    def drop(self, grid):
        if not self.carrying:
            return
        density = self.neighbourhood_density(grid)
        drop_prob = 1 - (DROP_SCALE * density / (1 + DROP_SCALE * density)) if self.is_anti_agent else (DROP_SCALE * density / (1 + DROP_SCALE * density))
        if random.random() < drop_prob:
            grid[self.x, self.y] += 1
            self.carrying = False

# Function to find the largest cluster
def find_largest_cluster(grid):
    visited = np.zeros_like(grid, dtype=bool)
    max_cluster_size = 0

    def dfs(x, y):
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return 0
        if visited[x, y] or grid[x, y] == 0:
            return 0
        visited[x, y] = True
        cluster_size = 1
        cluster_size += dfs(x+1, y)
        cluster_size += dfs(x-1, y)
        cluster_size += dfs(x, y+1)
        cluster_size += dfs(x, y-1)
        return cluster_size

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] > 0 and not visited[i, j]:
                cluster_size = dfs(i, j)
                max_cluster_size = max(max_cluster_size, cluster_size)

    return max_cluster_size

# Run multiple simulations and calculate the average largest cluster size
def run_simulation(num_simulations, anti_agent_ratio, pick_scale, drop_scale, sim_steps):
    global PICK_SCALE, DROP_SCALE, SIMULATION_STEPS
    PICK_SCALE, DROP_SCALE, SIMULATION_STEPS = pick_scale, drop_scale, sim_steps
    largest_clusters = []

    for _ in range(num_simulations):
        # Reinitialize grid and robots
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        for _ in range(NUM_OBJECTS):
            x, y = np.random.randint(0, GRID_SIZE, 2)
            grid[x, y] += 1
        robots = [Robot(is_anti_agent=(i < anti_agent_ratio * NUM_ROBOTS)) for i in range(NUM_ROBOTS)]

        # Run the simulation
        for _ in range(SIMULATION_STEPS):
            for robot in robots:
                robot.move()
                if robot.carrying:
                    robot.drop(grid)
                else:
                    robot.pick_up(grid)

        # Measure the performance
        largest_clusters.append(find_largest_cluster(grid))

    return np.mean(largest_clusters)

# Test different percentages of anti-agents
def test_different_ratios(num_simulations, ratios, pick_scale, drop_scale, sim_steps):
    performance_results = []
    for ratio in ratios:
        avg_largest_cluster = run_simulation(num_simulations, ratio, pick_scale, drop_scale, sim_steps)
        performance_results.append((ratio, avg_largest_cluster))
        print(f"Anti-Agent Ratio: {ratio}, Average Largest Cluster Size: {avg_largest_cluster}")
    return performance_results

# Define parameters for testing
anti_agent_ratios = [0.0, 0.05, 0.1, 0.15, 0.2]
num_simulations = 5

# Test different sets of parameters
parameter_sets = [
    (1.5, 2.0, 5000),  # Original parameters
    (1.2, 1.8, 5000),  # Slightly lower PICK_SCALE and DROP_SCALE
    (1.5, 2.0, 7000),  # Longer simulation time
    (1.8, 2.2, 5000),  # Slightly higher PICK_SCALE and DROP_SCALE
    (1.5, 2.0, 3000)   # Shorter simulation time
]

# Collect results
all_results = {}
for pick_scale, drop_scale, sim_steps in parameter_sets:
    key = f"PICK_SCALE={pick_scale}, DROP_SCALE={drop_scale}, SIMULATION_STEPS={sim_steps}"
    all_results[key] = test_different_ratios(num_simulations, anti_agent_ratios, pick_scale, drop_scale, sim_steps)

# Plot the results
plt.figure(figsize=(12, 8))
for params, results in all_results.items():
    ratios, cluster_sizes = zip(*results)
    plt.plot(ratios, cluster_sizes, marker='o', label=params)

plt.xlabel('Anti-Agent Ratio')
plt.ylabel('Average Largest Cluster Size')
plt.title('Performance of Swarm Clustering with Anti-Agents\nNUM_OBJECTS = {} | NUM_ROBOTS = {}'.format(NUM_OBJECTS, NUM_ROBOTS))
plt.legend()
plt.show()
