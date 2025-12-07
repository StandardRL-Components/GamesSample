def policy(env):
    # Strategy: Maximize tower coverage by prioritizing cannon placement at high-value intersections
    # during placement phase, and immediately start waves when no affordable placements remain.
    # Cannon towers provide better range and damage efficiency for gold cost in later waves.
    
    if env.game_phase == "wave":
        return [0, 0, 0]  # No actions during wave phase
    
    current_pos = env.cursor_pos
    empty_zones = [zone for zone in env.placement_zones 
                  if zone not in [t["pos"] for t in env.towers]]
    
    if not empty_zones or env.gold < 30:
        return [0, 0, 0]  # Start wave if no placements or insufficient gold
    
    if current_pos in empty_zones:
        # Build cannon if affordable in later waves, otherwise gun tower
        if env.gold >= 75 and env.current_wave >= 3:
            return [0, 0, 1]  # Build cannon
        elif env.gold >= 30:
            return [0, 1, 0]  # Build gun tower
    
    # Move to nearest empty zone using list order traversal
    current_idx = env.placement_zones.index(current_pos)
    empty_idxs = [i for i, zone in enumerate(env.placement_zones) if zone in empty_zones]
    if not empty_idxs:
        return [0, 0, 0]
    
    # Find closest empty zone in circular list
    dists = [(idx - current_idx) % len(env.placement_zones) for idx in empty_idxs]
    min_dist = min(dists)
    target_idx = empty_idxs[dists.index(min_dist)]
    
    if min_dist <= len(env.placement_zones) // 2:
        return [2, 0, 0] if target_idx > current_idx else [1, 0, 0]
    else:
        return [1, 0, 0] if target_idx > current_idx else [2, 0, 0]