def policy(env):
    # Strategy: Maximize forward speed while avoiding walls. Prioritize acceleration and controlled turning to complete laps quickly.
    # Use visual cues from the observation to detect walls and adjust steering accordingly.
    obs = env._get_observation()
    h, w, _ = obs.shape
    wall_color = np.array([255, 50, 50])
    tolerance = 50
    
    # Sample points ahead and to the sides
    points = [
        (h//4, w//2),      # Center ahead
        (h//4, w//2 - 50), # Left ahead
        (h//4, w//2 + 50)  # Right ahead
    ]
    
    # Check for wall proximity
    is_wall = []
    for y, x in points:
        color = obs[y, x]
        if np.all(np.abs(color - wall_color) < tolerance):
            is_wall.append(True)
        else:
            is_wall.append(False)
    
    center, left, right = is_wall
    
    # Determine action based on wall detection
    if center and not left and not right:
        return [2, 0, 0]  # Brake if wall directly ahead
    elif left and not right:
        return [4, 0, 1]  # Turn right with drift if left wall
    elif right and not left:
        return [3, 0, 1]  # Turn left with drift if right wall
    elif center and (left or right):
        return [2, 0, 0]  # Brake if surrounded
    else:
        return [1, 0, 1]  # Accelerate with drift by default