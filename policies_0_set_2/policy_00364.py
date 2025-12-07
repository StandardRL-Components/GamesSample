def policy(env):
    # Strategy: Push boxes toward their zones using Manhattan distance heuristic.
    # Prioritize moves that push boxes directly toward goals, avoiding stuck positions.
    # Use a1=0 and a2=0 since secondary actions have no effect in this environment.
    if env.game_over:
        return [0, 0, 0]
    
    player_x, player_y = env.player_pos
    best_action = 0
    best_score = -float('inf')
    
    for move in [1, 2, 3, 4]:  # up, down, left, right
        dx, dy = (0, -1) if move == 1 else (0, 1) if move == 2 else (-1, 0) if move == 3 else (1, 0)
        new_x, new_y = player_x + dx, player_y + dy
        
        if (new_x, new_y) in env.walls:
            continue
            
        box = next((b for b in env.boxes if b['pos'] == (new_x, new_y)), None)
        score = 0
        
        if box:
            push_x, push_y = new_x + dx, new_y + dy
            if (push_x, push_y) in env.walls or any(b['pos'] == (push_x, push_y) for b in env.boxes):
                continue  # Invalid push
                
            zone = next((z for z in env.zones if z['id'] == box['id']), None)
            if zone:
                current_dist = abs(box['pos'][0] - zone['pos'][0]) + abs(box['pos'][1] - zone['pos'][1])
                new_dist = abs(push_x - zone['pos'][0]) + abs(push_y - zone['pos'][1])
                score = current_dist - new_dist  # Positive if moving closer
                
                if new_dist == 0:
                    score += 10  # Bonus for reaching zone
                elif env._is_stuck({'pos': (push_x, push_y), 'id': box['id']}):
                    score -= 5  # Penalize stuck positions
        else:
            # Move without pushing: score by proximity to boxes not in zones
            for box in env.boxes:
                if not env._is_on_correct_zone(box):
                    dist = abs(new_x - box['pos'][0]) + abs(new_y - box['pos'][1])
                    score += 1 / (dist + 1)  # Closer to box is better
        
        if score > best_score:
            best_score = score
            best_action = move
    
    return [best_action, 0, 0]