def policy(env):
    # Strategy: Prioritize collecting coins while avoiding zombies. Move towards nearest coin if safe, 
    # otherwise move away from nearest zombie. Break ties by preferring movement over no-op.
    player_pos = env.player_pos
    zombies = [z['pos'] for z in env.zombies]
    coins = env.coins
    
    actions = [0, 1, 2, 3, 4]  # 0: no-op, 1: up, 2: down, 3: left, 4: right
    best_action = 0
    best_score = float('-inf')
    
    for a in actions:
        new_x, new_y = player_pos[0], player_pos[1]
        if a == 1: new_y -= 1
        elif a == 2: new_y += 1
        elif a == 3: new_x -= 1
        elif a == 4: new_x += 1
        
        new_x = max(0, min(env.GRID_SIZE-1, new_x))
        new_y = max(0, min(env.GRID_SIZE-1, new_y))
        new_pos = [new_x, new_y]
        
        score = 0
        # Reward collecting coins
        if new_pos in coins:
            score += 100
        # Penalize moving into zombies
        if new_pos in zombies:
            score -= 1000
        # Penalize adjacent zombies
        for z in zombies:
            if abs(new_x - z[0]) + abs(new_y - z[1]) == 1:
                score -= 10
        # Encourage movement towards coins if no immediate reward
        if new_pos not in coins and coins:
            min_dist = min(abs(new_x - c[0]) + abs(new_y - c[1]) for c in coins)
            score -= min_dist
        # Prefer movement over no-op
        if a == 0:
            score -= 1
            
        if score > best_score:
            best_score = score
            best_action = a
            
    return [best_action, 0, 0]