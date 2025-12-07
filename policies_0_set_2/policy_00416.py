def policy(env):
    # Strategy: Prioritize collecting gems while avoiding enemies. For each possible movement, 
    # compute a score based on proximity to gems (higher score for closer) and enemies (penalty for closer).
    # Choose the action with the highest score to maximize reward from gem collection and minimize life loss.
    player_x, player_y = env.player['x'], env.player['y']
    grid_w, grid_h = env.GRID_WIDTH, env.GRID_HEIGHT
    best_action, best_score = 0, -float('inf')
    
    for move in range(5):
        dx, dy = 0, 0
        if move == 1: dy = -1
        elif move == 2: dy = 1
        elif move == 3: dx = -1
        elif move == 4: dx = 1
        
        nx, ny = max(0, min(grid_w-1, player_x + dx)), max(0, min(grid_h-1, player_y + dy))
        score = 0.0
        
        # Reward proximity to gems
        for gem in env.gems:
            dist = abs(nx - gem['x']) + abs(ny - gem['y'])
            score += (1.0 / (dist + 1e-5)) * 10  # Closer gems get higher score
        
        # Penalty for proximity to enemies
        for enemy in env.enemies:
            dist = abs(nx - enemy['x']) + abs(ny - enemy['y'])
            score -= (1.0 / (dist + 1e-5)) * 20  # Closer enemies get higher penalty
            
        # Bonus for moving onto a gem
        if any(nx == gem['x'] and ny == gem['y'] for gem in env.gems):
            score += 50
            
        # Large penalty for moving onto an enemy
        if any(nx == enemy['x'] and ny == enemy['y'] for enemy in env.enemies):
            score -= 1000
            
        if score > best_score:
            best_score = score
            best_action = move
            
    return [best_action, 0, 0]