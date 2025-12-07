def policy(env):
    # Strategy: Prioritize attacking adjacent enemies when health is sufficient, otherwise defend.
    # Move towards stairs to advance levels, collecting nearby gold along the way.
    # This balances combat efficiency with progression for high reward.
    player_pos = env.player_pos
    stairs_pos = env.stairs_pos
    enemy_positions = [e['pos'] for e in env.enemies]
    if env.boss:
        enemy_positions.append(env.boss['pos'])
    
    # Check for adjacent enemies
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        adj_pos = (player_pos[0] + dx, player_pos[1] + dy)
        if adj_pos in enemy_positions:
            if env.player_health < 30:
                return [0, 0, 1]  # Defend if low health
            else:
                return [0, 1, 0]  # Attack otherwise
    
    # Check for adjacent gold
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        adj_pos = (player_pos[0] + dx, player_pos[1] + dy)
        if adj_pos in env.gold_pieces:
            move_dir = { (0,-1): 1, (0,1): 2, (-1,0): 3, (1,0): 4 }[(dx,dy)]
            return [move_dir, 0, 0]
    
    # Move towards stairs using Manhattan distance
    best_move = 0
    best_dist = float('inf')
    for move, (dx, dy) in enumerate([(0,0), (0,-1), (0,1), (-1,0), (1,0)], start=0):
        if move == 0:
            continue
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        if (0 <= new_pos[0] < env.GRID_WIDTH and 0 <= new_pos[1] < env.GRID_HEIGHT and 
            env.grid[new_pos[0]][new_pos[1]] == 0 and new_pos not in enemy_positions):
            dist = abs(new_pos[0] - stairs_pos[0]) + abs(new_pos[1] - stairs_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_move = move
                
    return [best_move, 0, 0]