def policy(env):
    # Strategy: Direct the ball toward the goal by placing gates that optimally adjust its trajectory.
    # Prioritize placing gates in the ball's path that maximize alignment with the goal direction.
    # Use available gate inventory efficiently, cycling selections only when necessary.
    
    if env.game_over:
        return [0, 0, 0]
    
    ball_x, ball_y = env.ball_grid_pos
    goal_x, goal_y = env.goal_pos
    vel_x, vel_y = env.ball_vel
    
    # Predict next ball position
    next_x = ball_x + vel_x
    next_y = ball_y + vel_y
    
    # If ball is moving toward goal, do nothing
    goal_dx = goal_x - ball_x
    goal_dy = goal_y - ball_y
    current_alignment = vel_x * goal_dx + vel_y * goal_dy
    if current_alignment > 0 and 0 <= next_x < env.GRID_SIZE and 0 <= next_y < env.GRID_SIZE:
        return [0, 0, 0]
    
    # Find best gate type to align with goal
    best_gate = None
    best_score = -float('inf')
    for gate_type in [env.GATE_OR, env.GATE_XOR, env.GATE_NOT, env.GATE_AND]:
        if env.gate_inventory[gate_type] <= 0:
            continue
        # Simulate gate effect
        if gate_type == env.GATE_OR:
            new_vel = [vel_y, -vel_x]
        elif gate_type == env.GATE_XOR:
            new_vel = [-vel_y, vel_x]
        elif gate_type == env.GATE_NOT:
            new_vel = [-vel_x, -vel_y]
        else:
            new_vel = [vel_x, vel_y]
        alignment = new_vel[0] * goal_dx + new_vel[1] * goal_dy
        if alignment > best_score:
            best_score = alignment
            best_gate = gate_type
    
    if best_gate is None:
        return [0, 0, 0]
    
    # Cycle gate selection if needed
    current_gate = env.gate_types[env.selected_gate_idx]
    if current_gate != best_gate:
        return [0, 0, 1]
    
    # Move cursor to target position (next ball cell)
    cursor_x, cursor_y = env.cursor_pos
    target_x, target_y = next_x, next_y
    
    # Generate movement action
    move_action = 0
    if cursor_x < target_x:
        move_action = 4
    elif cursor_x > target_x:
        move_action = 3
    elif cursor_y < target_y:
        move_action = 2
    elif cursor_y > target_y:
        move_action = 1
    
    # Place gate if cursor at target
    if cursor_x == target_x and cursor_y == target_y:
        if env.grid[target_y, target_x] == env.GATE_NONE and (target_x, target_y) != env.start_pos and (target_x, target_y) != env.goal_pos:
            return [0, 1, 0]
    
    return [move_action, 0, 0]