def policy(env):
    # Strategy: Prioritize moving held organ to target with correct rotation, then pick up nearest unplaced organ.
    # Maximizes reward by minimizing moves: +10 for correct placement, +100 for level completion.
    # Avoids penalties by moving directly to targets and only placing when rotation matches.
    
    # If game over, return no-op
    if env.game_over:
        return [0, 0, 0]
    
    # If holding an organ, move it to target position and handle rotation/placement
    if env.held_organ_idx is not None:
        organ = env.organs[env.held_organ_idx]
        target_pos = organ["target_pos"]
        current_pos = organ["pos"]
        
        # If at target position with correct rotation, place it
        if current_pos == target_pos and organ["rotation"] == organ["target_rotation"]:
            return [0, 0, 1]
        
        # If at target position but wrong rotation, rotate
        if current_pos == target_pos:
            return [0, 1, 0]
        
        # Move toward target position using Manhattan distance
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        
        if abs(dx) > abs(dy):
            return [4 if dx > 0 else 3, 0, 0]
        else:
            return [2 if dy > 0 else 1, 0, 0]
    
    # Find nearest unplaced organ
    unplaced_organs = [o for o in env.organs if not o["is_placed"]]
    if not unplaced_organs:
        return [0, 0, 0]
    
    # Calculate Manhattan distances to cursor
    distances = [
        (abs(o["pos"].x - env.cursor_pos.x) + abs(o["pos"].y - env.cursor_pos.y), i)
        for i, o in enumerate(unplaced_organs)
    ]
    _, nearest_idx = min(distances)
    target_organ = unplaced_organs[nearest_idx]
    
    # If cursor is on organ, pick it up
    if env.cursor_pos == target_organ["pos"]:
        return [0, 1, 0]
    
    # Move toward nearest unplaced organ
    dx = target_organ["pos"].x - env.cursor_pos.x
    dy = target_organ["pos"].y - env.cursor_pos.y
    
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dy > 0 else 1, 0, 0]