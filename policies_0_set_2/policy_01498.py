def policy(env):
    # This policy uses a simple heuristic to maximize line clears while minimizing risk:
    # 1. Always hard drop (a1=1) to maximize placement speed and line clears per second
    # 2. Use left/right movements (a0=3/4) to center pieces for better stacking
    # 3. Avoid rotations (a0=1, a2=1) unless necessary to prevent inefficient placements
    # This balances rapid piece placement with basic positioning to achieve the 20-line target before time runs out.
    current_x = env.current_piece["x"]
    piece_width = len(env.current_piece["shape"][env.current_piece["rotation"]][0])
    target_x = (env.GRID_WIDTH - piece_width) // 2
    
    if current_x < target_x:
        return [4, 1, 0]  # Move right + hard drop
    elif current_x > target_x:
        return [3, 1, 0]  # Move left + hard drop
    else:
        return [0, 1, 0]  # Hard drop immediately