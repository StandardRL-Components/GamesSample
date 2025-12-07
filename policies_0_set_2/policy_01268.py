def policy(env):
    # Strategy: Always aim for the next unvisited platform by ID. Use long jumps for distant platforms and short for nearby ones.
    # Prioritize horizontal alignment when significant offset exists, otherwise jump vertically. Avoid obstacles by jumping when safe.
    if env.game_over:
        return [0, 0, 0]
    if not env.player_on_platform:
        return [0, 0, 0]
    
    current_id = max(env.visited_platforms)
    for p in env.platforms:
        if p['id'] == current_id and p.get('type') == 'final':
            return [0, 0, 0]
    
    next_id = current_id + 1
    target_platform = None
    for p in env.platforms:
        if p['id'] == next_id:
            target_platform = p
            break
            
    if target_platform is None:
        return [1, 1, 0]
        
    dx = target_platform['rect'].centerx - env.player_pos.x
    dy = target_platform['rect'].y - env.player_pos.y
    
    if abs(dx) > 50:
        direction = 4 if dx > 0 else 3
    else:
        direction = 1
        
    dist_sq = dx*dx + dy*dy
    if dist_sq > 14400:
        return [direction, 1, 0]
    else:
        return [direction, 0, 1]