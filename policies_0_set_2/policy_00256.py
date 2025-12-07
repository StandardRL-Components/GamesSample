def policy(env):
    # Strategy: Prioritize attacking adjacent enemies to eliminate threats and gain rewards,
    # then move toward health pickups if health is low, otherwise navigate toward exit.
    # Avoid walls and enemies when moving. Break ties by preferring exit direction.
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    enemies = [e['pos'] for e in env.enemies if e['health'] > 0]
    pickups = env.pickups
    player_health = env.player_health
    player_facing = env.player_facing
    dungeon_map = env.dungeon_map

    # Check attack opportunity in current facing direction
    attack_pos = player_pos + player_facing
    if any((attack_pos == enemy).all() for enemy in enemies):
        return [0, 1, 0]

    # Evaluate movement candidates
    best_score = -float('inf')
    best_action = [0, 0, 0]
    moves = [(0, [0,0]), (1, [0,-1]), (2, [0,1]), (3, [-1,0]), (4, [1,0])]
    
    for move_code, move_dir in moves:
        if move_code == 0:
            continue  # Skip no-move since we already checked attack
        new_pos = player_pos + move_dir
        if dungeon_map[new_pos[1], new_pos[0]] == 1:
            continue  # Skip walls

        # Calculate score for this move
        score = 0
        # Prefer exit
        exit_dist = abs(exit_pos[0] - new_pos[0]) + abs(exit_pos[1] - new_pos[1])
        score -= exit_dist * 0.5
        # Prefer health pickups if health is low
        if player_health < env.PLAYER_MAX_HEALTH * 0.6:
            for pickup in pickups:
                if (new_pos == pickup).all():
                    score += 10
                    break
        # Avoid enemies
        for enemy in enemies:
            if (new_pos == enemy).all():
                score -= 100
            elif abs(new_pos[0] - enemy[0]) + abs(new_pos[1] - enemy[1]) == 1:
                score -= 5

        if score > best_score:
            best_score = score
            best_action = [move_code, 0, 0]

    return best_action