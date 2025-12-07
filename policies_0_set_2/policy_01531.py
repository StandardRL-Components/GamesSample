def policy(env):
    # Strategy: Maximize upward movement (rewarded) by jumping toward center (x=15) when safe, avoiding enemies by jumping away.
    # Prioritizes y-increase jumps (up-left/up-right) to reach top platform (+100 reward) while evading enemies (-20 penalty) and falling.
    if not env.player_on_ground or env.player_jump_cooldown > 0:
        return [0, 0, 0]
    
    current_platform = None
    current_platform_idx = -1
    for idx, plat in enumerate(env.platforms):
        if (plat['pos'][0] <= env.player_pos[0] <= plat['pos'][0] + plat['w'] and
            plat['pos'][1] <= env.player_pos[1] <= plat['pos'][1] + plat['h'] and
            abs(env.player_pos[2] - (plat['pos'][2] + plat['depth'])) < 0.1):
            current_platform = plat
            current_platform_idx = idx
            break
            
    if current_platform is None:
        return [0, 0, 0]
        
    if current_platform['type'] == 'top':
        return [0, 0, 0]
        
    enemy_on_platform = None
    for enemy in env.enemies:
        if enemy['platform_idx'] == current_platform_idx:
            enemy_on_platform = enemy
            break
            
    if env.player_pos[0] <= 0:
        return [0, 1, 0]
    if env.player_pos[0] >= 30:
        return [1, 0, 0]
        
    if enemy_on_platform is not None:
        if enemy_on_platform['pos'][0] < env.player_pos[0]:
            return [0, 1, 0]
        else:
            return [1, 0, 0]
            
    if env.player_pos[0] > 15:
        return [1, 0, 0]
    else:
        return [0, 1, 0]