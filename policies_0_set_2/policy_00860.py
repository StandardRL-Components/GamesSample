def policy(env):
    # Strategy: Use Manhattan distance heuristic to push blocks toward targets. 
    # Simulate each direction to minimize total distance, breaking ties by direction priority (up, down, left, right).
    # Avoid no-ops by checking if blocks move. Return no-op during animations or if game is over.
    if env.is_pushing or env.game_over:
        return [0, 0, 0]
    
    def compute_heuristic(blocks, targets):
        total = 0
        for i, block in enumerate(blocks):
            tx, ty = targets[i]['pos']
            bx, by = block
            total += abs(bx - tx) + abs(by - ty)
        return total
    
    def simulate_push(blocks, move, grid_size):
        cols, rows = grid_size
        blocks = [list(b) for b in blocks]
        occupied = set(tuple(b) for b in blocks)
        
        if move == 1: dx, dy, sort_key, rev = 0, -1, 1, False
        elif move == 2: dx, dy, sort_key, rev = 0, 1, 1, True
        elif move == 3: dx, dy, sort_key, rev = -1, 0, 0, False
        elif move == 4: dx, dy, sort_key, rev = 1, 0, 0, True
        else: return blocks
        
        sorted_blocks = sorted(blocks, key=lambda b: b[sort_key], reverse=rev)
        new_blocks = [None] * len(blocks)
        occupied_temp = occupied.copy()
        
        for i, block in enumerate(sorted_blocks):
            x, y = block
            while True:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < cols and 0 <= ny < rows):
                    break
                if (nx, ny) in occupied_temp:
                    break
                x, y = nx, ny
            new_pos = (x, y)
            occupied_temp.discard(tuple(block))
            occupied_temp.add(new_pos)
            new_blocks[i] = new_pos
        
        return new_blocks
    
    current_blocks = [tuple(b['pos']) for b in env.blocks]
    current_heuristic = compute_heuristic(current_blocks, env.targets)
    best_move = 0
    best_heuristic = current_heuristic
    
    for move in [1, 2, 3, 4]:
        new_blocks = simulate_push(current_blocks, move, (env.GRID_COLS, env.GRID_ROWS))
        if new_blocks == current_blocks:
            continue
        new_heuristic = compute_heuristic(new_blocks, env.targets)
        if new_heuristic < best_heuristic:
            best_heuristic = new_heuristic
            best_move = move
    
    return [best_move, 0, 0]