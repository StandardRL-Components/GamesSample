def policy(env):
    # Strategy: Prioritize survival by avoiding zombies and collecting ammo when low.
    # Shoot zombies when safe, using shift to auto-collect ammo when critically low.
    # Move away from nearby zombies, toward ammo when needed, and break ties consistently.
    
    # Read current state
    player_pos = env.player_pos
    ammo = env.player_ammo
    zombies = env.zombies
    ammo_drops = env.ammo_drops
    cooldown = env.shoot_cooldown
    
    # Find nearest zombie and its distance
    min_zombie_dist = float('inf')
    nearest_zombie = None
    for z in zombies:
        dist = player_pos.distance_to(z['pos'])
        if dist < min_zombie_dist:
            min_zombie_dist = dist
            nearest_zombie = z
    
    # Find nearest ammo drop if exists
    nearest_ammo = None
    if ammo_drops:
        nearest_ammo = min(ammo_drops, key=lambda a: player_pos.distance_to(a['pos']))
    
    # Determine movement action
    move_action = 0  # Default: no movement
    if nearest_zombie and min_zombie_dist < 100:  # Zombie too close - flee
        flee_dir = player_pos - nearest_zombie['pos']
        if abs(flee_dir.x) > abs(flee_dir.y):
            move_action = 4 if flee_dir.x > 0 else 3
        else:
            move_action = 2 if flee_dir.y > 0 else 1
    elif ammo < 20 and nearest_ammo:  # Low ammo - seek ammo
        seek_dir = nearest_ammo['pos'] - player_pos
        if abs(seek_dir.x) > abs(seek_dir.y):
            move_action = 4 if seek_dir.x > 0 else 3
        else:
            move_action = 2 if seek_dir.y > 0 else 1
    elif nearest_zombie and min_zombie_dist > 200:  # Zombie far - approach
        approach_dir = nearest_zombie['pos'] - player_pos
        if abs(approach_dir.x) > abs(approach_dir.y):
            move_action = 4 if approach_dir.x > 0 else 3
        else:
            move_action = 2 if approach_dir.y > 0 else 1
    
    # Determine shoot action (only if cooldown ready and ammo available)
    shoot_action = 1 if cooldown == 0 and ammo > 0 and nearest_zombie and min_zombie_dist < 250 else 0
    
    # Determine shift action (auto-seek ammo when critically low)
    shift_action = 1 if ammo < 10 and ammo_drops else 0
    
    return [move_action, shoot_action, shift_action]