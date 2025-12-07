def policy(env):
    # Strategy: Guide rider to finish by drawing upward ramps to boost rightward velocity.
    # Shift is held to reduce bounce and maintain momentum. Draw lines when velocity is low
    # and cursor is positioned to create a favorable ramp. Prioritize forward progress.
    rider_x, rider_y = env.rider_pos
    rider_vx = env.rider_vel[0]
    finish_x = env.finish_x
    
    # Calculate desired cursor position (right and above rider)
    desired_x = rider_x + 100
    desired_y = rider_y - 50
    desired_x = max(0, min(env.WIDTH, desired_x))
    desired_y = max(0, min(env.HEIGHT, desired_y))
    
    # Move cursor toward desired position
    dx = desired_x - env.cursor_pos[0]
    dy = desired_y - env.cursor_pos[1]
    if abs(dx) > 15 or abs(dy) > 15:
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 1 if dy < 0 else 2
    else:
        a0 = 0
    
    # Draw line if velocity is low and not at finish
    a1 = 1 if (rider_vx < 5 and not env.prev_space_held and rider_x < finish_x - 10) else 0
    
    # Always hold shift for controlled movement
    a2 = 1
    
    return [a0, a1, a2]