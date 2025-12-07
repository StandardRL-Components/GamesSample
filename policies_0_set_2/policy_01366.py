def policy(env):
    # Strategy: Move towards the nearest soul while avoiding next-step specter positions.
    # Prioritize immediate soul collection, then minimize distance to next soul. Avoid moves that lead to specter collisions.
    current_pos = env.player_pos
    next_specter_positions = set()
    for specter in env.specters:
        next_index = (specter['path_index'] + 1) % len(specter['path'])
        next_specter_positions.add(specter['path'][next_index])
    
    if env.souls:
        nearest_soul = min(env.souls, key=lambda s: abs(s[0]-current_pos[0]) + abs(s[1]-current_pos[1]))
    else:
        nearest_soul = None
    
    moves = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
    safe_actions = []
    for a0, (dx, dy) in enumerate(moves):
        new_x = max(0, min(env.GRID_W-1, current_pos[0] + dx))
        new_y = max(0, min(env.GRID_H-1, current_pos[1] + dy))
        if (new_x, new_y) not in next_specter_positions:
            safe_actions.append(a0)
    
    if safe_actions:
        if nearest_soul:
            best_action = min(safe_actions, key=lambda a0: (
                abs((current_pos[0] + moves[a0][0]) - nearest_soul[0]) + 
                abs((current_pos[1] + moves[a0][1]) - nearest_soul[1])
            ))
            return [best_action, 0, 0]
        return [safe_actions[0], 0, 0]
    
    if nearest_soul:
        best_action = min(range(5), key=lambda a0: (
            abs((current_pos[0] + moves[a0][0]) - nearest_soul[0]) + 
            abs((current_pos[1] + moves[a0][1]) - nearest_soul[1])
        ))
        return [best_action, 0, 0]
    return [0, 0, 0]