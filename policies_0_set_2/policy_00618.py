def policy(env):
    """
    Maximizes survival by moving away from the nearest enemy. Computes safe moves that avoid collisions 
    and selects the direction with the highest alignment to the escape vector. Returns no movement only 
    if no safe moves exist to avoid immediate collision.
    """
    player_pos = env.player_pos
    enemies = env.enemies
    player_size = env.PLAYER_SIZE
    screen_w, screen_h = env.SCREEN_WIDTH, env.SCREEN_HEIGHT
    
    # Find nearest enemy
    min_dist_sq = float('inf')
    nearest_enemy = None
    for enemy in enemies:
        dx = player_pos.x - enemy['pos'].x
        dy = player_pos.y - enemy['pos'].y
        dist_sq = dx*dx + dy*dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_enemy = enemy
    
    if not nearest_enemy:
        return [1, 0, 0]  # Default to up if no enemies
    
    # Compute escape vector from nearest enemy
    escape_dx = player_pos.x - nearest_enemy['pos'].x
    escape_dy = player_pos.y - nearest_enemy['pos'].y
    
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
    actions = [1, 2, 3, 4]
    safe_actions = []
    
    # Check each move for safety (no collision)
    for idx, (dx, dy) in enumerate(moves):
        new_x = player_pos.x + dx * env.PLAYER_SPEED
        new_y = player_pos.y + dy * env.PLAYER_SPEED
        new_x = max(0, min(screen_w - player_size, new_x))
        new_y = max(0, min(screen_h - player_size, new_y))
        rect = (int(new_x), int(new_y), player_size, player_size)
        
        safe = True
        for enemy in enemies:
            closest_x = max(rect[0], min(enemy['pos'].x, rect[0] + rect[2]))
            closest_y = max(rect[1], min(enemy['pos'].y, rect[1] + rect[3]))
            dist_sq = (closest_x - enemy['pos'].x)**2 + (closest_y - enemy['pos'].y)**2
            if dist_sq < enemy['radius']**2:
                safe = False
                break
                
        if safe:
            safe_actions.append(actions[idx])
    
    # Select best safe move aligned with escape direction
    if safe_actions:
        best_action = None
        best_dot = -float('inf')
        for action in safe_actions:
            move_vec = moves[action - 1]
            dot = escape_dx * move_vec[0] + escape_dy * move_vec[1]
            if dot > best_dot:
                best_dot = dot
                best_action = action
        return [best_action, 0, 0]
    
    return [0, 0, 0]  # No safe moves