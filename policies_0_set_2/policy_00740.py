def policy(env):
    # Strategy: Maintain forward momentum by aligning track with sled velocity to minimize crashes.
    # Adjust track angle to match velocity direction for smooth riding, using shorter segments when
    # corrections are needed. Prioritize keeping sled on player-drawn track for speed boosts.
    import math
    
    # Get current sled velocity and calculate its angle
    vx, vy = env.sled_vel
    velocity_angle = math.atan2(vy, vx) if abs(vx) > 0.1 or abs(vy) > 0.1 else env.last_draw_angle
    
    # Calculate angle error between current track direction and velocity direction
    angle_error = velocity_angle - env.last_draw_angle
    
    # Choose movement action to minimize angle error
    possible_angles = [
        env.last_draw_angle,  # no-op
        env.last_draw_angle - 0.2,  # up
        env.last_draw_angle + 0.2,  # down
        env.last_draw_angle + 0.4,  # left (sharp down)
        env.last_draw_angle - 0.4   # right (sharp up)
    ]
    best_action = min(range(5), key=lambda a: abs(possible_angles[a] - velocity_angle))
    
    # Use shorter segments when making significant corrections for better control
    shift_held = 1 if abs(angle_error) > 0.3 else 0
    
    return [best_action, 0, shift_held]