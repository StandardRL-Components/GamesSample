def policy(env):
    """
    Prioritize shooting (a1=1) when cooldown allows to destroy aliens for immediate rewards.
    Avoid incoming projectiles by moving horizontally away from threats.
    When safe, align horizontally with the lowest alien (highest threat) to maximize hit probability.
    Secondary action (a2) is unused, so always 0.
    """
    if env.game_over:
        return [0, 0, 0]
    
    player_x, player_y = env.player_pos
    a1 = 1 if env.player_shoot_cooldown == 0 else 0
    a2 = 0
    
    # Check for immediate projectile threats
    danger_zone = 50
    closest_proj = None
    min_dist = float('inf')
    for proj in env.alien_projectiles:
        dy = player_y - proj[1]
        if 0 <= dy <= danger_zone and abs(proj[0] - player_x) < 20:
            if dy < min_dist:
                min_dist = dy
                closest_proj = proj
                
    if closest_proj:
        a0 = 4 if closest_proj[0] < player_x else 3
    else:
        # Target lowest alien (highest y-value) above player
        target_x = None
        max_y = -1
        for alien in env.aliens:
            if alien['pos'][1] < player_y and alien['pos'][1] > max_y:
                max_y = alien['pos'][1]
                target_x = alien['pos'][0]
                
        if target_x is None:
            a0 = 0
        else:
            if player_x < target_x - 5:
                a0 = 4
            elif player_x > target_x + 5:
                a0 = 3
            else:
                a0 = 0
                
    return [a0, a1, a2]