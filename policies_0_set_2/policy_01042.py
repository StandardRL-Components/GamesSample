def policy(env):
    """
    Strategy: Prioritize catching fruits (especially special ones) while avoiding bombs. Evaluate each possible movement
    by scoring immediate fruit catches (5 for special, 1 for normal) and penalizing bomb collisions (-1000) and being under
    bombs (-1). Choose the highest-scoring action, breaking ties by preferring non-no-op actions to avoid the no-op penalty.
    """
    player_pos = env.player_pos
    fruits = env.fruits
    bombs = env.bombs
    fall_speed = env.fall_speed
    
    actions = [0, 1, 2, 3, 4]  # no-op, up, down, left, right
    best_score = -float('inf')
    best_action = 0
    
    for action in actions:
        # Calculate candidate position
        candidate_pos = player_pos.copy()
        if action == 1:  # up
            candidate_pos[1] = max(0, candidate_pos[1] - 1)
        elif action == 2:  # down
            candidate_pos[1] = min(env.GRID_HEIGHT - 1, candidate_pos[1] + 1)
        elif action == 3:  # left
            candidate_pos[0] = max(0, candidate_pos[0] - 1)
        elif action == 4:  # right
            candidate_pos[0] = min(env.GRID_WIDTH - 1, candidate_pos[0] + 1)
        
        score = 0
        # Penalize no-op
        if action == 0:
            score -= 0.1
        
        # Check for fruit catches
        for fruit in fruits:
            next_y = fruit['pos'][1] + fall_speed
            if (int(fruit['pos'][0]) == candidate_pos[0] and 
                int(next_y) == candidate_pos[1]):
                score += 5 if fruit['type'] == 'special' else 1
        
        # Check for bomb collisions and penalties
        for bomb in bombs:
            next_bomb_y = bomb['pos'][1] + fall_speed
            # Bomb collision (lose life)
            if (int(bomb['pos'][0]) == candidate_pos[0] and 
                int(next_bomb_y) == candidate_pos[1]):
                score -= 1000
            # Being under bomb penalty
            if (int(bomb['pos'][0]) == candidate_pos[0] and 
                candidate_pos[1] < next_bomb_y):
                score -= 1.0
        
        # Update best action
        if score > best_score or (score == best_score and action != 0):
            best_score = score
            best_action = action
    
    return [best_action, 0, 0]