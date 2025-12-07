def policy(env):
    """
    Maximizes reward by prioritizing survival (dodging enemy bullets) while aggressively targeting aliens.
    Strategy: 1) Dodge imminent enemy bullets by moving horizontally away from threats.
              2) Align horizontally with nearest alien to enable shooting.
              3) Continuously fire when possible to maximize alien destruction and stage progression.
    """
    if env.game_over:
        return [0, 0, 0]
    
    player_x, player_y = env.player_pos
    
    # Check for imminent enemy bullet threats (within 30px vertically and 20px horizontally)
    threat_bullets = []
    for bullet in env.enemy_bullets:
        if bullet['pos'][1] < player_y:  # Bullet above player
            dy = player_y - bullet['pos'][1]
            dx = abs(bullet['pos'][0] - player_x)
            if dy < 30 and dx < 20:
                threat_bullets.append((dy, bullet))
    
    # Prioritize dodging imminent threats
    if threat_bullets:
        closest_bullet = min(threat_bullets, key=lambda x: x[0])[1]
        if closest_bullet['pos'][0] < player_x:
            move_action = 4  # Move right if threat is left
        else:
            move_action = 3  # Move left if threat is right
    else:
        # Target nearest alien above player
        candidate_aliens = [a for a in env.aliens if a['pos'][1] < player_y]
        if candidate_aliens:
            closest_alien = min(candidate_aliens, key=lambda a: abs(a['pos'][0] - player_x))
            dx = closest_alien['pos'][0] - player_x
            if abs(dx) > 5:  # Only move if significantly misaligned
                move_action = 3 if dx < 0 else 4
            else:
                move_action = 0
        else:
            move_action = 0
    
    # Always fire when possible (cooldown expired and bullet limit not reached)
    fire_action = 1 if (env.player_fire_timer == 0 and 
                       len(env.player_bullets) < env.MAX_PLAYER_BULLETS) else 0
    
    return [move_action, fire_action, 0]