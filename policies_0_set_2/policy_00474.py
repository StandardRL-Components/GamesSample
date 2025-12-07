def policy(env):
    # Strategy: Prioritize mining nearest asteroid while avoiding enemies. Move towards closest ore-rich asteroid when safe, 
    # mine when in range (a1=1), and evade enemies by moving away when they're too close. Use brake (a0=2) to slow down when 
    # needed for precise mining. Always maintain safe distance from enemies while maximizing ore collection.
    import numpy as np
    
    player_pos = env.player_pos
    danger_radius = 150.0
    mining_range = 100.0
    
    # Find nearest enemy
    enemy_distances = [np.linalg.norm(player_pos - enemy['pos']) for enemy in env.enemies]
    nearest_enemy_dist = min(enemy_distances) if enemy_distances else float('inf')
    nearest_enemy_idx = np.argmin(enemy_distances) if enemy_distances else None
    
    # Find nearest asteroid with ore
    asteroid_dists = []
    asteroid_positions = []
    for ast in env.asteroids:
        if ast['ore'] > 0:
            dist = np.linalg.norm(player_pos - ast['pos'])
            asteroid_dists.append(dist)
            asteroid_positions.append(ast['pos'])
    nearest_ast_dist = min(asteroid_dists) if asteroid_dists else float('inf')
    nearest_ast_idx = np.argmin(asteroid_dists) if asteroid_dists else None
    
    # Evade enemies if too close
    if nearest_enemy_dist < danger_radius:
        enemy_pos = env.enemies[nearest_enemy_idx]['pos']
        evade_dir = player_pos - enemy_pos
        evade_dir /= np.linalg.norm(evade_dir)
        
        # Choose movement direction away from enemy
        if abs(evade_dir[0]) > abs(evade_dir[1]):
            a0 = 3 if evade_dir[0] < 0 else 4
        else:
            a0 = 1 if evade_dir[1] < 0 else 2
        return [a0, 0, 0]
    
    # Move toward nearest asteroid if beyond mining range
    if nearest_ast_dist > mining_range and nearest_ast_idx is not None:
        ast_pos = asteroid_positions[nearest_ast_idx]
        move_dir = ast_pos - player_pos
        move_dir /= np.linalg.norm(move_dir)
        
        # Choose movement direction toward asteroid
        if abs(move_dir[0]) > abs(move_dir[1]):
            a0 = 3 if move_dir[0] < 0 else 4
        else:
            a0 = 1 if move_dir[1] < 0 else 2
        return [a0, 0, 0]
    
    # Mine when close to asteroid and safe from enemies
    if nearest_ast_dist <= mining_range:
        return [0, 1, 0]
    
    # Default: no movement or action
    return [0, 0, 0]