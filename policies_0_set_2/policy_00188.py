def policy(env):
    # Strategy: Accelerate continuously and steer towards the nearest track point to stay centered.
    # Use boost whenever available to maximize speed and finish quickly. Avoid braking to maintain momentum.
    movement = 1  # Accelerate by default
    boost = 1 if env.boost_cooldown == 0 else 0  # Use boost if off cooldown
    brake = 0  # Never brake to maintain speed
    
    # Get nearest track point and steer toward it to stay centered
    _, closest_point = env._is_on_track()
    if closest_point is not None:
        dx = closest_point[0] - env.world_offset[0]  # X difference in world coordinates
        if dx < -5:  # Track is left, steer left
            movement = 3
        elif dx > 5:  # Track is right, steer right
            movement = 4
    
    return [movement, boost, brake]