def policy(env):
    # Strategy: Use current robot and exit positions to compute Manhattan distance. 
    # Move in the direction that minimizes distance to exit, prioritizing horizontal movement 
    # since exit is horizontally aligned. This greedy approach efficiently navigates toward the goal.
    robot_x, robot_y = env.robot_pos
    exit_x, exit_y = env.exit_pos
    dx = exit_x - robot_x
    dy = exit_y - robot_y
    
    if dx > 0:
        return [4, 0, 0]  # Move right
    elif dx < 0:
        return [3, 0, 0]  # Move left
    elif dy > 0:
        return [2, 0, 0]  # Move down
    elif dy < 0:
        return [1, 0, 0]  # Move up
    else:
        return [0, 0, 0]  # No movement (already at exit)