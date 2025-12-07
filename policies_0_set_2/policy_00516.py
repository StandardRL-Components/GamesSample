def policy(env):
    # Strategy: Maintain safe distance while prioritizing shooting zombies when facing them. 
    # Reload only when out of ammo and not under immediate threat. Use kiting: move away while shooting.
    # Avoid close contact to prevent health loss while maximizing zombie elimination rewards.
    if not env.zombies:
        return [0, 0, 0]
    
    player_pos = env.player_pos
    closest_zombie = min(env.zombies, key=lambda z: (z['pos'][0] - player_pos[0])**2 + (z['pos'][1] - player_pos[1])**2)
    zombie_pos = closest_zombie['pos']
    dx = zombie_pos[0] - player_pos[0]
    dy = zombie_pos[1] - player_pos[1]
    dist_sq = dx*dx + dy*dy
    
    # Calculate direction vector dot product for facing check
    player_dir = env.player_last_move_dir
    dot = dx*player_dir[0] + dy*player_dir[1]
    
    # Determine movement direction (away from closest zombie if too close)
    move_dir = 0
    if dist_sq < 2500:  # 50 units squared
        if abs(dx) > abs(dy):
            move_dir = 4 if dx < 0 else 3
        else:
            move_dir = 1 if dy < 0 else 2
    
    # Shooting and reload logic
    shoot = 0
    reload = 0
    
    if not env.player_is_reloading:
        if env.player_ammo == 0:
            reload = 1
        elif env.player_shoot_cooldown_timer == 0 and dot > 0 and dist_sq < 10000:  # 100 units squared
            shoot = 1
    
    return [move_dir, shoot, reload]