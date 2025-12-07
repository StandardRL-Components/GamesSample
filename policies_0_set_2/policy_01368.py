def policy(env):
    """Strategy: Prioritize moving towards exit while avoiding rats. Score actions based on distance reduction and rat proximity, using sprint when safe to maximize progress."""
    # Movement mapping: 0=none, 1=up, 2=down, 3=left, 4=right
    moves = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    rats = env.rat_positions
    
    best_score = -float('inf')
    best_action = [0, 0, 0]  # Default no-op
    
    # Evaluate all movement actions with and without sprint
    for move_dir in range(1, 5):
        dx, dy = moves[move_dir]
        for sprint in [0, 1]:
            # Calculate new position after movement
            new_pos = player_pos
            steps = 2 if sprint else 1
            for _ in range(steps):
                next_pos = (new_pos[0] + dx, new_pos[1] + dy)
                if env._is_valid_path(next_pos):
                    new_pos = next_pos
                else:
                    break  # Stop if hit wall
            
            # Skip if no movement occurred
            if new_pos == player_pos:
                continue
                
            # Calculate score components
            dist_before = env._manhattan_distance(player_pos, exit_pos)
            dist_after = env._manhattan_distance(new_pos, exit_pos)
            dist_score = dist_before - dist_after  # Positive if closer
            
            # Penalize proximity to rats
            rat_penalty = 0
            for rat in rats:
                d = env._manhattan_distance(new_pos, rat)
                if d == 0:
                    rat_penalty += 100  # Immediate collision
                elif d == 1:
                    rat_penalty += 10   # Adjacent to rat
            
            # Combine scores (prioritize distance reduction, avoid rats)
            score = dist_score * 5 - rat_penalty
            
            if score > best_score:
                best_score = score
                best_action = [move_dir, 0, sprint]
    
    return best_action