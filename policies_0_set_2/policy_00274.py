def policy(env):
    """
    Greedy push policy for Sokoban: prioritize moves that push blocks toward targets, 
    avoid increasing distance. For empty moves, move toward nearest block not on target.
    Secondary actions unused, set to 0.
    """
    if env.game_over:
        return [0, 0, 0]
    
    player_pos = env.player_pos
    block_positions = env.block_positions
    target_positions = env.target_positions
    on_target_blocks = env._get_on_target_blocks()
    
    move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    best_score = -float('inf')
    best_action = 0
    
    for a0 in range(5):
        if a0 == 0:
            score = -1.0
        else:
            dx, dy = move_map[a0]
            new_player = (player_pos[0] + dx, player_pos[1] + dy)
            if not (0 <= new_player[0] < 16 and 0 <= new_player[1] < 10):
                score = -10.0
            elif new_player in block_positions:
                block_idx = block_positions.index(new_player)
                if block_idx in on_target_blocks:
                    score = -5.0
                else:
                    new_block = (new_player[0] + dx, new_player[1] + dy)
                    if not (0 <= new_block[0] < 16 and 0 <= new_block[1] < 10) or new_block in block_positions:
                        score = -10.0
                    else:
                        old_dist = abs(new_player[0] - target_positions[block_idx][0]) + abs(new_player[1] - target_positions[block_idx][1])
                        new_dist = abs(new_block[0] - target_positions[block_idx][0]) + abs(new_block[1] - target_positions[block_idx][1])
                        if new_dist < old_dist:
                            score = 10.0
                        elif new_dist == old_dist:
                            score = 0.0
                        else:
                            score = -10.0
            else:
                min_dist = float('inf')
                for i, block in enumerate(block_positions):
                    if i not in on_target_blocks:
                        dist = abs(new_player[0] - block[0]) + abs(new_player[1] - block[1])
                        min_dist = min(min_dist, dist)
                score = -min_dist / 10.0
        
        if score > best_score:
            best_score = score
            best_action = a0
    
    return [best_action, 0, 0]