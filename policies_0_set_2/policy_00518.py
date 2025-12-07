def policy(env):
    """
    This policy uses a greedy one-step lookahead to minimize the total Manhattan distance of crates to goals.
    It prioritizes moves that push crates onto goals, then moves that reduce overall crate-goal distance.
    If no improving move is found, it moves the player towards the nearest crate not on a goal.
    """
    if env.game_over:
        return [0, 0, 0]
    
    player = env.player_pos
    crates = env.crate_positions
    goals = env.goal_positions
    walls = env.wall_positions
    
    if all(crate in goals for crate in crates):
        return [0, 0, 0]
    
    best_score = float('inf')
    best_move = 0
    wall_set = walls
    crate_set = set(crates)
    
    for move in [1, 2, 3, 4]:
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move]
        new_player = (player[0] + dx, player[1] + dy)
        
        if new_player in wall_set:
            continue
            
        new_crates = list(crates)
        if new_player in crate_set:
            new_crate_pos = (new_player[0] + dx, new_player[1] + dy)
            if new_crate_pos in wall_set or new_crate_pos in crate_set:
                continue
            idx = new_crates.index(new_player)
            new_crates[idx] = new_crate_pos
        
        total_dist = 0
        for crate in new_crates:
            if crate in goals:
                dist = 0
            else:
                dist = min(abs(crate[0] - g[0]) + abs(crate[1] - g[1]) for g in goals)
            total_dist += dist
            
        min_player_dist = 0
        if total_dist > 0:
            non_goal_crates = [c for c in new_crates if c not in goals]
            if non_goal_crates:
                min_player_dist = min(abs(new_player[0] - c[0]) + abs(new_player[1] - c[1]) for c in non_goal_crates)
        
        score = total_dist * 1000 + min_player_dist
        
        if score < best_score:
            best_score = score
            best_move = move
            
    return [best_move, 0, 0]