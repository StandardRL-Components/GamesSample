def policy(env):
    # Aim at lowest available blocks first to maximize breaks and minimize ball loss.
    # Adjust launcher angle to target bottom row blocks, launch when aligned.
    if env.ball_state != "ready" or env.game_over or not env.blocks:
        return [0, 0, 0]
    
    # Find lowest blocks (highest y-coordinate)
    max_y = max(block['rect'].bottom for block in env.blocks)
    lowest_blocks = [b for b in env.blocks if b['rect'].bottom == max_y]
    
    # Target leftmost block among lowest to encourage horizontal spread
    target_block = min(lowest_blocks, key=lambda b: b['rect'].centerx)
    target_x = target_block['rect'].centerx
    
    # Calculate desired angle (scaled by screen width)
    dx = target_x - env.launcher_pos.x
    desired_angle = 80 * (dx / (env.WIDTH / 2))
    desired_angle = max(-80, min(80, desired_angle))
    
    # Adjust launcher or launch
    tolerance = 1.5
    if abs(env.launcher_angle - desired_angle) <= tolerance:
        return [0, 1, 0]
    elif env.launcher_angle < desired_angle:
        return [3, 0, 0]
    else:
        return [4, 0, 0]