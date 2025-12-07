def policy(env):
    """
    Follows the track centerline to stay within safe bounds and hit checkpoints.
    Scans the column at the car's fixed x-position to detect track boundaries by
    finding top/bottom blue pixels (track color). Moves toward the center between
    these bounds to avoid crashes and align with checkpoints (which are on the centerline).
    Uses a dead zone to prevent oscillation.
    """
    obs = env._get_observation()
    car_x = env.WIDTH // 4
    top_bound = None
    bottom_bound = None
    
    # Scan column at car's x-position to find track boundaries (blue pixels)
    for y in range(env.HEIGHT):
        if obs[y, car_x, 2] > 100:  # Blue channel threshold for track
            top_bound = y
            break
    for y in range(env.HEIGHT - 1, -1, -1):
        if obs[y, car_x, 2] > 100:
            bottom_bound = y
            break
            
    if top_bound is None or bottom_bound is None:
        return [0, 0, 0]  # No track detected, do nothing
    
    center = (top_bound + bottom_bound) / 2
    error = center - env.car_y
    
    # Dead zone to prevent oscillation
    if abs(error) < 5:
        return [0, 0, 0]
    return [2 if error > 0 else 1, 0, 0]