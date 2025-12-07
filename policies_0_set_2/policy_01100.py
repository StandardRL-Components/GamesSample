def policy(env):
    # Strategy: Prioritize attacking adjacent monsters when cooldown allows for immediate rewards.
    # If no attack is possible, move towards nearest monster to set up future attacks while avoiding unnecessary moves.
    # Ignore secondary action (a2) as it's unused in this environment.
    player_pos = env.player_grid_pos
    monsters = [m['grid_pos'] for m in env.monsters]
    
    # Check if we can attack now (cooldown ready and adjacent monster exists)
    adjacent_monster = any(abs(m[0]-player_pos[0]) + abs(m[1]-player_pos[1]) == 1 for m in monsters)
    attack = 1 if env.player_attack_cd == 0 and adjacent_monster else 0
    
    # Handle movement based on cooldown and strategy
    if env.player_move_cd > 0:
        move = 0
    else:
        if not monsters:
            move = 0
        else:
            # Find nearest monster
            min_dist = float('inf')
            nearest = None
            for m in monsters:
                dist = abs(m[0]-player_pos[0]) + abs(m[1]-player_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest = m
            
            # Move toward nearest monster if not adjacent, else hold position
            if min_dist > 1:
                dx = nearest[0] - player_pos[0]
                dy = nearest[1] - player_pos[1]
                if abs(dx) > abs(dy):
                    move = 4 if dx > 0 else 3
                else:
                    move = 2 if dy > 0 else 1
            else:
                move = 0
    
    return [move, attack, 0]