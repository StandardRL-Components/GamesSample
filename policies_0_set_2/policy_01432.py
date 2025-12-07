def policy(env):
    """
    Strategy: Plant immediately if current cell has sufficient resources, otherwise move to the best empty cell
    based on minimum resource levels. Prioritizes planting Vibra-Tulips (type 0) due to higher score potential
    and faster growth, but will plant Giga-Ferns if resources favor them. Avoids planting on occupied cells.
    """
    x, y = env.cursor_pos
    current_type = env.selected_plant_type
    plant_costs = [p['cost'] for p in env.PLANT_TYPES]
    
    # Check if we can plant at current position
    if env.grid[y][x] is None:
        # Check resources for both plant types
        res = env.cell_resources[y][x]
        can_plant_current = (res['water'] >= plant_costs[current_type]['water'] and 
                            res['nutrients'] >= plant_costs[current_type]['nutrients'])
        alt_type = 1 - current_type
        can_plant_alt = (res['water'] >= plant_costs[alt_type]['water'] and 
                        res['nutrients'] >= plant_costs[alt_type]['nutrients'])
        
        if can_plant_current:
            return [0, 1, 0]  # Plant current type
        elif can_plant_alt:
            return [0, 1, 1]  # Switch type and plant
    
    # Find best movement direction to empty cell with highest minimum resources
    best_score = -1
    best_action = 0  # Default: no movement
    directions = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]  # (dx, dy, action)
    
    for dx, dy, action in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < env.GRID_W and 0 <= ny < env.GRID_H:
            if env.grid[ny][nx] is None:
                res = env.cell_resources[ny][nx]
                score = min(res['water'], res['nutrients'])
                if score > best_score:
                    best_score = score
                    best_action = action
    
    return [best_action, 0, 0]  # Move to best cell, no plant/toggle