def policy(env):
    # Strategy: Align current block with monsters at lowest positions to maximize crushing and prevent game over.
    # Prioritize dropping on monsters, especially those near bottom, and avoid safe drops without monsters.
    if env.current_block is None:
        return [0, 0, 0]
    
    # Helper function to compute landing y for hypothetical block
    def get_landing_y(block_type, x0):
        shape = env.BLOCK_SHAPES[block_type]
        for y_offset in range(env.GRID_HEIGHT):
            for dx, dy in shape:
                x, y = x0 + dx, y_offset + dy
                if y >= env.GRID_HEIGHT or (0 <= x < env.GRID_WIDTH and env.grid[x, y] != 0):
                    return y_offset - 1
        return env.GRID_HEIGHT - 1 - max(dy for _, dy in shape)
    
    best_reward = -10
    best_action = [0, 0, 0]
    best_crushed = -1
    best_no_drop_action = [0, 0, 0]
    
    for cycle in [0, 1]:
        block_type = (env.current_block['type'] + cycle) % 3
        shape = env.BLOCK_SHAPES[block_type]
        min_x = -min(dx for dx, _ in shape)
        max_x = env.GRID_WIDTH - 1 - max(dx for dx, _ in shape)
        
        for move in [0, 3, 4]:
            x0 = env.current_block['x']
            if move == 3:
                x0 -= 1
            elif move == 4:
                x0 += 1
            x0 = max(min_x, min(x0, max_x))
            
            landing_y = get_landing_y(block_type, x0)
            crushed = 0
            for dx, dy in shape:
                x, y = x0 + dx, landing_y + dy
                for m in env.monsters:
                    if m['x'] == x and m['y'] == y:
                        crushed += 1
                        break
            
            if crushed > 0:
                reward = crushed + (5 + crushed - 1 if crushed > 1 else 0)
                if reward > best_reward:
                    best_reward = reward
                    best_action = [move, 1, cycle]
            elif crushed > best_crushed:
                best_crushed = crushed
                best_no_drop_action = [move, 0, cycle]
    
    return best_action if best_reward > 0 else best_no_drop_action