def policy(env):
    """
    Maximizes reward by prioritizing survival through strategic dodging of enemy projectiles,
    collecting health powerups when low, and aggressively targeting nearest enemy while maintaining
    optimal firing distance. Uses simplified threat assessment and deterministic movement choices.
    """
    player = env.player
    player_pos = player['pos']
    
    # Dodge nearby enemy projectiles with priority
    for proj in env.enemy_projectiles:
        dist = player_pos.distance_to(proj['pos'])
        if dist < 50:
            dx = player_pos.x - proj['pos'].x
            dy = player_pos.y - proj['pos'].y
            if abs(dx) > abs(dy):
                return [4 if dx > 0 else 3, 1 if player['shoot_cooldown'] <= 0 else 0, 0]
            else:
                return [2 if dy > 0 else 1, 1 if player['shoot_cooldown'] <= 0 else 0, 0]
    
    # Collect health powerup if health is low
    if player['health'] < 50:
        for pwr in env.powerups:
            if pwr['type'] == 'health':
                dx = pwr['pos'].x - player_pos.x
                dy = pwr['pos'].y - player_pos.y
                if abs(dx) > abs(dy):
                    movement = 4 if dx > 0 else 3
                else:
                    movement = 2 if dy > 0 else 1
                return [movement, 1 if player['shoot_cooldown'] <= 0 else 0, 0]
    
    # Target nearest enemy while maintaining distance
    if env.enemies:
        nearest = min(env.enemies, key=lambda e: player_pos.distance_to(e['pos']))
        dx = nearest['pos'].x - player_pos.x
        dy = nearest['pos'].y - player_pos.y
        dist = player_pos.distance_to(nearest['pos'])
        
        # Maintain optimal firing distance (~100 pixels)
        if dist > 150:
            if abs(dx) > abs(dy):
                movement = 4 if dx > 0 else 3
            else:
                movement = 2 if dy > 0 else 1
        elif dist < 80:
            if abs(dx) > abs(dy):
                movement = 3 if dx > 0 else 4
            else:
                movement = 1 if dy > 0 else 2
        else:
            movement = 0  # Hold position at optimal distance
        return [movement, 1 if player['shoot_cooldown'] <= 0 else 0, 0]
    
    # Default: minimal movement with continuous firing
    return [0, 1 if player['shoot_cooldown'] <= 0 else 0, 0]