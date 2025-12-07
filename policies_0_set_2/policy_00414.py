def policy(env):
    """
    One-step lookahead policy that simulates each directional move and selects the one with the highest immediate reward.
    Prioritizes moves that lock blocks (+5 reward) or reduce Manhattan distance to targets (+1 reward). Avoids moves that
    increase distance (-1 reward) or waste moves when no positive reward is available (no-op). Secondary actions are unused.
    """
    if env.game_over or all(block['locked'] for block in env.blocks):
        return [0, 0, 0]
    
    def simulate_move(movement):
        if movement == 1:
            dx, dy = (0, -1)
            sort_key = lambda b: b['pos'][1]
            reverse = False
        elif movement == 2:
            dx, dy = (0, 1)
            sort_key = lambda b: b['pos'][1]
            reverse = True
        elif movement == 3:
            dx, dy = (-1, 0)
            sort_key = lambda b: b['pos'][0]
            reverse = False
        else:
            dx, dy = (1, 0)
            sort_key = lambda b: b['pos'][0]
            reverse = True
            
        occupied = {tuple(block['pos']) for block in env.blocks}
        movable_blocks = [b for b in env.blocks if not b['locked']]
        sorted_blocks = sorted(movable_blocks, key=sort_key, reverse=reverse)
        reward = 0
        
        for block in sorted_blocks:
            old_pos = tuple(block['pos'])
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)
            if (0 <= new_pos[0] < env.GRID_COLS and 0 <= new_pos[1] < env.GRID_ROWS) and new_pos not in occupied:
                occupied.remove(old_pos)
                occupied.add(new_pos)
                target = next(t for t in env.targets if t['color'] == block['color'])
                old_dist = abs(old_pos[0]-target['pos'][0]) + abs(old_pos[1]-target['pos'][1])
                new_dist = abs(new_pos[0]-target['pos'][0]) + abs(new_pos[1]-target['pos'][1])
                if new_dist < old_dist:
                    reward += 1
                elif new_dist > old_dist:
                    reward -= 1
                if new_pos == target['pos']:
                    reward += 5
        return reward

    best_reward = -10**9
    best_move = 0
    for move in [1, 2, 3, 4]:
        reward = simulate_move(move)
        if reward > best_reward:
            best_reward = reward
            best_move = move
            
    return [best_move, 0, 0] if best_reward > 0 else [0, 0, 0]