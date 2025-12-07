def policy(env):
    # Strategy: Maximize reward by placing affordable towers in free slots covering path waypoints.
    # Prioritize Machine Guns for cost efficiency, then Cannons for splash damage in later waves.
    # Move selection to nearest free slot and cycle to affordable tower type if needed.
    current_slot = env.selected_tower_slot_idx
    current_type = env.selected_tower_type_idx
    tower_cost = env.tower_types[current_type]['cost']
    slot_pos = env.tower_slots[current_slot]
    is_slot_free = not any(tower.pos == slot_pos for tower in env.towers)
    
    if is_slot_free and env.gold >= tower_cost:
        return [0, 1, 0]  # Place tower if slot free and affordable
    
    # Find nearest free slot by index order
    free_slots = [i for i in range(8) if not any(t.pos == env.tower_slots[i] for t in env.towers)]
    if free_slots:
        target_slot = free_slots[0]
        current_row, current_col = current_slot // 4, current_slot % 4
        target_row, target_col = target_slot // 4, target_slot % 4
        
        if current_row != target_row:
            return [2 if current_row < target_row else 1, 0, 0]  # Move vertically
        elif current_col != target_col:
            return [4 if current_col < target_col else 3, 0, 0]  # Move horizontally
    
    # Cycle to affordable tower type if current unaffordable
    if env.gold < tower_cost:
        affordable_types = [i for i, t in enumerate(env.tower_types) if env.gold >= t['cost']]
        if affordable_types and affordable_types[0] != current_type:
            return [0, 0, 1]  # Cycle to next type
    
    return [0, 0, 0]  # No action otherwise