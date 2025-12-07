def policy(env):
    # Strategy: Prioritize attacking adjacent enemies for immediate rewards, then collect gold, then move toward boss.
    # Avoid dangerous positions when health is low. Break ties by moving toward boss to maximize long-term reward.
    movement_delta = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    
    # Attack if adjacent enemy exists
    if env._find_nearest_target() is not None:
        return [0, 1, 0]
    
    player_pos = env.player['pos']
    
    # Stay to collect gold if on gold pile
    for gold in env.gold_piles:
        if gold['pos'] == player_pos:
            return [0, 0, 0]
    
    best_score = -10**9
    best_action = 0
    
    # Evaluate each movement direction
    for action in range(5):
        dx, dy = movement_delta[action]
        new_pos = [player_pos[0] + dx, player_pos[1] + dy]
        if not env._is_valid_and_walkable(new_pos):
            continue
            
        score = 0
        # Reward moving toward boss
        score -= env._manhattan_distance(new_pos, env.boss['pos']) * 2
        
        # Reward collecting gold
        for gold in env.gold_piles:
            if gold['pos'] == new_pos:
                score += 10
                
        # Penalize dangerous positions
        for enemy in env.enemies:
            dist = env._manhattan_distance(new_pos, enemy['pos'])
            if dist <= 1:
                score -= 100 if env.player['health'] < 30 else 10
                
        if env.boss['health'] > 0:
            dist_boss = env._manhattan_distance(new_pos, env.boss['pos'])
            if dist_boss <= 1:
                score -= 200 if env.player['health'] < 30 else 20
            elif dist_boss <= 3:
                score -= 50 if env.player['health'] < 50 else 5
                
        if score > best_score:
            best_score = score
            best_action = action
            
    return [best_action, 0, 0]