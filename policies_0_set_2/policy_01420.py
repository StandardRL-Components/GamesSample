def policy(env):
    # Greedy strategy: prioritize moves towards finish line (right) while avoiding obstacles and collecting boosts.
    # Maximizes reward by minimizing moves (avoid obstacles) and collecting boosts for immediate reward.
    current_x, current_y = env.snail_pos
    best_score = -10**9
    best_action = 0  # Default to no movement

    # Evaluate possible moves in priority order: right, up, down, left, none
    for a0 in [4, 1, 2, 3, 0]:
        dx, dy = 0, 0
        if a0 == 1: dy = -1  # Up
        elif a0 == 2: dy = 1  # Down
        elif a0 == 3: dx = -1  # Left
        elif a0 == 4: dx = 1  # Right

        next_x = max(0, min(env.GRID_SIZE-1, current_x + dx))
        next_y = max(0, min(env.GRID_SIZE-1, current_y + dy))
        
        # Skip obstacle moves
        if (next_x, next_y) in env.obstacles:
            continue
            
        # Score: prioritize right movement (x-coordinate) and boost collection
        score = next_x * 10  # Strong bias toward right movement
        if (next_x, next_y) in env.boosts:
            score += 50  # Bonus for boosts
            
        if score > best_score:
            best_score = score
            best_action = a0
            
    return [best_action, 0, 0]  # a1 and a2 unused in this environment