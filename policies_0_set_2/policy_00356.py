def policy(env):
    # Use direct pixel color analysis to navigate towards goal while avoiding walls.
    # Prioritize moving right towards goal, with vertical adjustments based on goal position.
    # Check adjacent pixels for walls to avoid invalid moves and conserve energy.
    obs = env._get_observation()
    h, w, _ = obs.shape
    
    # Find robot position (cyan: (0,200,255))
    robot_pixels = []
    for y in range(50, h-50):  # Skip UI areas
        for x in range(50, w-50):
            r, g, b = obs[y, x]
            if r < 50 and g > 180 and b > 200:  # Robot color
                robot_pixels.append((x, y))
    if not robot_pixels:
        return [0, 0, 0]
    rx = sum(p[0] for p in robot_pixels) // len(robot_pixels)
    ry = sum(p[1] for p in robot_pixels) // len(robot_pixels)
    
    # Find goal position (green: (0,255,150))
    goal_pixels = []
    for y in range(50, h-50):
        for x in range(50, w-50):
            r, g, b = obs[y, x]
            if r < 50 and g > 200 and 100 < b < 200:  # Goal color
                goal_pixels.append((x, y))
    if not goal_pixels:
        return [0, 0, 0]
    gx = sum(p[0] for p in goal_pixels) // len(goal_pixels)
    gy = sum(p[1] for p in goal_pixels) // len(goal_pixels)
    
    # Estimate cell size from robot pixel spread
    xs = [p[0] for p in robot_pixels]
    cell_size = max(1, (max(xs) - min(xs)) // 2)
    
    # Check adjacent cells for walls (gray: (80,90,100))
    def is_wall(x, y):
        if 0 <= x < w and 0 <= y < h:
            r, g, b = obs[y, x]
            return 70 <= r <= 90 and 80 <= g <= 100 and 90 <= b <= 110
        return True
    
    # Prioritize horizontal movement towards goal
    dx = gx - rx
    dy = gy - ry
    actions = []
    if dx > 0 and not is_wall(rx + cell_size, ry):
        actions.append(4)  # Right
    elif dx < 0 and not is_wall(rx - cell_size, ry):
        actions.append(3)  # Left
    
    # Then vertical movement towards goal
    if dy < 0 and not is_wall(rx, ry - cell_size):
        actions.append(1)  # Up
    elif dy > 0 and not is_wall(rx, ry + cell_size):
        actions.append(2)  # Down
    
    # Fallback to any valid move if goal-directed moves are blocked
    if not actions:
        for move, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)], 1):
            if not is_wall(rx + dx * cell_size, ry + dy * cell_size):
                actions.append(move)
    
    return [actions[0] if actions else 0, 0, 0]