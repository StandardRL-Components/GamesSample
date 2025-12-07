def policy(env):
    # Strategy: Move towards nearest gem while avoiding nearby enemies. Prioritize gem collection for high reward.
    # Calculate squared distance between two points
    def sq_dist(a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2

    player_pos = env.player_pos
    gems = [gem['pos'] for gem in env.gems]
    enemies = [enemy['pos'] for enemy in env.enemies]
    
    # Find nearest gem
    if gems:
        nearest_gem = min(gems, key=lambda g: sq_dist(player_pos, g))
        gem_dir = nearest_gem - player_pos
    else:
        gem_dir = np.array([0, 0])
    
    # Check for nearby enemies (within 2 units)
    nearby_enemies = [e for e in enemies if sq_dist(player_pos, e) < 4]
    if nearby_enemies:
        # Move away from nearest enemy
        nearest_enemy = min(nearby_enemies, key=lambda e: sq_dist(player_pos, e))
        avoid_dir = player_pos - nearest_enemy
        move_x = 2 if avoid_dir[0] > 0 else (3 if avoid_dir[0] < 0 else 0)
        move_y = 1 if avoid_dir[1] > 0 else (2 if avoid_dir[1] < 0 else 0)
        action = move_x if abs(avoid_dir[0]) > abs(avoid_dir[1]) else move_y
    else:
        # Move toward nearest gem
        move_x = 4 if gem_dir[0] > 0 else (3 if gem_dir[0] < 0 else 0)
        move_y = 2 if gem_dir[1] > 0 else (1 if gem_dir[1] < 0 else 0)
        action = move_x if abs(gem_dir[0]) > abs(gem_dir[1]) else move_y
    
    return [action, 0, 0]