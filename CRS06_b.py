import numpy as np
import random
import matplotlib.pyplot as plt
#defining the grid size
GRID_SIZE = 30 

# Define the number of the robots and the anti-agents
no_of_robots = 100
no_of_anti_agents = 10

# defining maximum numbers of steps to simulate
maximum_steps = 1000

#defining the probability threshold for movement
probability_move = 0.1

grid = np.zeros((GRID_SIZE,GRID_SIZE))

#placing robots randomly on the grid
robots = []
for _ in range(no_of_robots):
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    grid[x, y] += 1
    robots.append((x,y))

#placing anti agents randomly
anti_agents = []
for _ in range(no_of_anti_agents):
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    anti_agents.append((x, y))

# Defining neighbourhood size
Neighbourhood_size = 1

def get_neighbourhood(x, y):
    neighbourhood = []
    for i in range(-Neighbourhood_size, Neighbourhood_size+1):
        for j in range(-Neighbourhood_size, Neighbourhood_size+1):
            if i == 0 and j == 0:
                continue
            nx, ny = (x + i) % GRID_SIZE, (y + j) % GRID_SIZE
            neighbourhood.append((nx, ny))
    return neighbourhood

def robot_on_move(x,y):
    direction = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
    nx, ny = (x + direction[0]) % GRID_SIZE, (y + direction[1]) % GRID_SIZE
    return nx, ny

def count_neighbourhood(x, y):
    return sum(grid[nx, ny] for nx, ny in get_neighbourhood(x, y))

# Simulate the swarm behavior
for step in range(maximum_steps):
    new_robots = []
    for x, y in robots:
        if random.random() < probability_move:
            grid[x, y] -= 1
            nx, ny = robot_on_move(x, y)
            grid[nx, ny] += 1
            new_robots.append((nx, ny))
        else:
            new_robots.append((x, y))

    new_anti_agents = []
    for x, y in anti_agents:
        neighbourhood = get_neighbourhood(x, y)
        for nx, ny in neighbourhood:
            if grid[nx, ny] > 0:
                if count_neighbourhood(nx, ny) > 2:  # Threshold to leave the cluster
                    grid[nx, ny] -= 1
                    mx, my = robot_on_move(nx, ny)
                    grid[mx, my] += 1
                    if (nx, ny) in new_robots:
                        new_robots.remove((nx, ny))
                    new_robots.append((mx, my))
        ax, ay = robot_on_move(x, y)
        new_anti_agents.append((ax, ay))

    robots = new_robots
    anti_agents = new_anti_agents

    # Optional: Visualize the grid (requires additional libraries like matplotlib)
    if step % 100 == 0:
        plt.imshow(grid, cmap='hot', interpolation='nearest')
        plt.title(f"Step {step}")
        plt.show()

# Final positions of robots and anti-agents
print("Final positions of robots:")
print(robots)

print("Final positions of anti-agents:")
print(anti_agents)