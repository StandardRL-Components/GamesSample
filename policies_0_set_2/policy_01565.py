def policy(env):
    """
    Strategy: Prioritize slicing fruits currently within range to maximize score and combo bonuses.
    If no sliceable fruits, move towards the lowest (most urgent) fruit to prevent misses and life loss.
    Avoid unnecessary movement to minimize penalty for fruits on screen.
    """
    # Check for any fruit within slicing range of current slicer position
    for fruit in env.fruits:
        if abs(fruit['pos'][1] - env.slicer_y) < fruit['type_info']['radius']:
            return [0, 1, 0]  # No movement, slice immediately

    # If no fruits, wait without moving or slicing
    if not env.fruits:
        return [0, 0, 0]

    # Find the lowest (most urgent) fruit to intercept
    lowest_fruit = max(env.fruits, key=lambda f: f['pos'][1])
    target_y = lowest_fruit['pos'][1]

    # Move towards target fruit's y-position
    if env.slicer_y < target_y:
        return [2, 0, 0]  # Move down
    else:
        return [1, 0, 0]  # Move up