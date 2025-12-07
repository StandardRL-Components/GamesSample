def policy(env):
    # Strategy: Track the rider's position and velocity to build optimal track segments.
    # Prioritize keeping the rider on track with slight downward slopes for speed,
    # using boosts on straight sections, and adjusting height to match the rider's trajectory.
    rider_y = env.rider_pos.y
    last_y = env.last_track_endpoint.y
    rider_on_ground = env.rider_on_ground
    steps = env.steps
    
    # Choose slope direction to align track with rider height
    if rider_y < last_y - 20:
        a0 = 1  # Up slope to reach higher rider
    elif rider_y > last_y + 20:
        a0 = 2  # Down slope to catch lower rider
    else:
        a0 = 0  # Straight (optimal for speed and boosts)
    
    # Use boost pads on straight sections periodically to maximize speed
    a2 = 1 if (steps % 8 == 0) and rider_on_ground and a0 == 0 else 0
    # Rarely use jumps to avoid destabilizing; focus on continuous track
    a1 = 1 if (steps % 40 == 0) and rider_on_ground else 0
    
    return [a0, a1, a2]