def policy(env):
    """
    Maximizes reward by prioritizing building affordable towers at optimal positions to defend against waves.
    Strategy: Move cursor to nearest empty build site, build if affordable, else wait for resources.
    Toggles tower type if current is unaffordable but other is affordable. Avoids no-ops by checking state.
    """
    # Get current state
    current_idx = env.cursor_index
    current_site = env.build_sites[current_idx]
    current_tower = env.TOWER_TYPES[env.selected_tower_type]
    resources = env.resources
    
    # Check if current site is occupied
    occupied = any(t['pos'] == current_site for t in env.towers)
    
    # Check affordability for both tower types
    affordable_types = [i for i, t in enumerate(env.TOWER_TYPES) if resources >= t['cost']]
    can_afford_current = env.selected_tower_type in affordable_types
    
    # Build if possible at current site
    if not occupied and can_afford_current:
        return [0, 1, 0]  # Build current tower
    
    # Find all empty sites
    empty_sites = [i for i, site in enumerate(env.build_sites) 
                  if not any(t['pos'] == site for t in env.towers)]
    
    if not empty_sites:
        return [0, 0, 0]  # No available sites
    
    # Move toward nearest empty site if current is occupied or unaffordable
    if occupied or not can_afford_current:
        target_idx = empty_sites[0]
        current_row, current_col = current_idx // 4, current_idx % 4
        target_row, target_col = target_idx // 4, target_idx % 4
        
        if target_row > current_row:
            move = 2  # Down
        elif target_row < current_row:
            move = 1  # Up
        elif target_col > current_col:
            move = 4  # Right
        elif target_col < current_col:
            move = 3  # Left
        else:
            move = 0  # Already at target
        
        # Toggle tower type if needed (current unaffordable but other affordable)
        if not can_afford_current and affordable_types:
            return [move, 0, 1]
        return [move, 0, 0]
    
    return [0, 0, 0]  # Default no-op