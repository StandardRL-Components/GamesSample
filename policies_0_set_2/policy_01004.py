def policy(env):
    # Strategy: Prioritize building towers in optimal path coverage order (near turns and base) with best affordable type.
    # Focus on early wave defense with type1, then transition to type2/3 for later waves and group damage.
    # Maximizes enemy kills and wave clears while protecting base health.
    desired_order = [0, 4, 3, 7, 1, 5, 2, 6]  # Zone priority: near turns and base first
    unoccupied = [i for i, zone in enumerate(env.tower_zones) if not zone["occupied"]]
    if not unoccupied:
        return [0, 0, 0]
    
    # Select target zone: highest priority unoccupied
    target_zone = next((idx for idx in desired_order if idx in unoccupied), unoccupied[0])
    current_idx = env.selected_zone_idx
    
    # Move to target zone if not selected
    if current_idx != target_zone:
        n = len(env.tower_zones)
        steps_right = (target_zone - current_idx) % n
        steps_left = (current_idx - target_zone) % n
        return [4, 0, 0] if steps_right <= steps_left else [3, 0, 0]
    
    # Determine best affordable tower type
    resources = env.resources
    if env.wave_number <= 2:
        desired_type = 1 if resources >= 30 else None
    else:
        if resources >= 60:
            desired_type = 3
        elif resources >= 40:
            desired_type = 2
        elif resources >= 30:
            desired_type = 1
        else:
            desired_type = None
    
    if desired_type is None:
        return [0, 0, 0]
    if env.selected_tower_type != desired_type:
        return [0, 0, 1]
    return [0, 1, 0]