def policy(env):
    """
    Strategy: Focus on powering machines by adjusting controllable gears to their target speeds. 
    For each controllable gear, compute the required speed to power its connected machines (considering gear ratios).
    Prioritize the gear with the highest error from its target, then adjust rotation accordingly.
    Avoid overloads by staying within safe speed limits.
    """
    # Get current state
    selected_idx = env.selected_gear_internal_idx
    controllable_gears = env.controllable_gear_indices
    gear0 = env.gears[controllable_gears[0]]
    gear3 = env.gears[controllable_gears[1]]
    
    # Calculate desired speeds for controllable gears to power all machines
    # Gear0 should drive gears 1 and 2 to ±4.5: optimal is average (0) but we must choose one target
    # Gear3 should drive gears 4 and 5 to ∓6.0: optimal is average (0) but we must choose one target
    # We prioritize powering unpowered machines by selecting direction that minimizes error
    target0 = 3.0 if not env.machines[1]['powered'] else -3.0  # Power machine1 (gear2) or machine0 (gear1)
    target3 = -4.0 if not env.machines[3]['powered'] else 4.0  # Power machine3 (gear5) or machine2 (gear4)
    
    # Calculate errors
    error0 = gear0['target_speed'] - target0
    error3 = gear3['target_speed'] - target3
    
    # Select gear with largest error
    if abs(error0) > abs(error3):
        target_gear = 0
        error = error0
    else:
        target_gear = 1
        error = error3
    
    # Determine action based on selected gear and error
    if target_gear != selected_idx:
        # Switch to the gear that needs adjustment
        movement = 2 if target_gear > selected_idx else 1
    else:
        # Adjust rotation to reduce error
        if abs(error) < 0.5:
            movement = 0  # No adjustment needed
        else:
            movement = 3 if error > 0 else 4  # Rotate left to decrease speed, right to increase
    
    return [movement, 0, 0]