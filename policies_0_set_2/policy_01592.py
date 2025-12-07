def policy(env):
    # Strategy: Simulate each directional push to maximize immediate reward by prioritizing moves that lock blocks (+5),
    # then moves that reduce Manhattan distance to targets (+0.5). Avoid moves that increase distance (-0.2) or are no-ops.
    # Break ties by direction priority (right, down, left, up) to ensure determinism and avoid oscillation.
    def simulate_direction(dx, dy):
        occupied = {b['pos'] for b in env.blocks if b['locked']}
        movable = [b for b in env.blocks if not b['locked']]
        sorted_blocks = sorted(movable, key=lambda b: b['pos'][0]*dx + b['pos'][1]*dy, reverse=(dx>0 or dy>0))
        moved = {}
        current_occupied = set(occupied)
        reward = 0.0
        
        for block in sorted_blocks:
            pos = block['pos']
            new_pos = (pos[0] + dx, pos[1] + dy)
            if (0 <= new_pos[0] < env.GRID_WIDTH and 0 <= new_pos[1] < env.GRID_HEIGHT 
                and new_pos not in current_occupied):
                moved[block['id']] = new_pos
                current_occupied.discard(pos)
                current_occupied.add(new_pos)
                old_dist = abs(pos[0]-block['target_pos'][0]) + abs(pos[1]-block['target_pos'][1])
                new_dist = abs(new_pos[0]-block['target_pos'][0]) + abs(new_pos[1]-block['target_pos'][1])
                if new_pos == block['target_pos']:
                    reward += 5.0
                elif new_dist < old_dist:
                    reward += 0.5
                elif new_dist > old_dist:
                    reward -= 0.2
        return reward

    best_dir, best_reward = 0, -10**9
    for direction, (dx, dy) in enumerate([(0,-1), (0,1), (-1,0), (1,0)], 1):
        reward = simulate_direction(dx, dy)
        if reward > best_reward:
            best_reward = reward
            best_dir = direction
    return [best_dir, 0, 0]