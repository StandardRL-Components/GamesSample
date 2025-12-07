def policy(env):
    """
    Greedy policy that simulates each push direction to maximize immediate reward.
    Chooses the direction that minimizes total Manhattan distance to target center
    and maximizes pixels in zone, avoiding moves that don't change state (no-ops).
    """
    def simulate_push(positions, direction):
        dx, dy = {1: (0,-1), 2: (0,1), 3: (-1,0), 4: (1,0)}[direction]
        new_pos = positions.copy()
        occupied = set(positions)
        n = len(positions)
        
        indices = list(range(n))
        if dx != 0:
            indices.sort(key=lambda i: positions[i][0], reverse=dx>0)
        else:
            indices.sort(key=lambda i: positions[i][1], reverse=dy>0)
            
        for i in indices:
            old = new_pos[i]
            candidate = (old[0] + dx, old[1] + dy)
            if (0 <= candidate[0] < env.GRID_SIZE and 
                0 <= candidate[1] < env.GRID_SIZE and 
                candidate not in occupied):
                occupied.remove(old)
                occupied.add(candidate)
                new_pos[i] = candidate
                
        return new_pos

    current_pos = [p['pos'] for p in env.pixels]
    current_in_zone = sum(env._is_in_target(p) for p in current_pos)
    current_dist = sum(env._manhattan_distance(p, env.target_center) for p in current_pos)
    
    best_reward = -float('inf')
    best_dir = 0
    
    for direction in [1,2,3,4]:
        new_pos = simulate_push(current_pos, direction)
        if new_pos == current_pos:
            continue
            
        new_in_zone = sum(env._is_in_target(p) for p in new_pos)
        new_dist = sum(env._manhattan_distance(p, env.target_center) for p in new_pos)
        
        reward = (new_in_zone - current_in_zone) * 1.0 + (current_dist - new_dist) * 0.1
        if reward > best_reward:
            best_reward = reward
            best_dir = direction
            
    return [best_dir, 0, 0] if best_reward > -float('inf') else [0, 0, 0]