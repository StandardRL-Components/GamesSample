def policy(env):
    """
    This policy uses a greedy approach to maximize immediate rewards by pushing blocks towards their targets.
    It simulates each possible push direction (up, down, left, right) to determine which move locks the most blocks
    onto their targets. If multiple moves yield the same number of locks, it prioritizes moves that actually move blocks
    to avoid no-ops. If no beneficial move is found, it returns no action to avoid wasting moves.
    """
    if env.game_state == "ANIMATING" or env.game_over:
        return [0, 0, 0]
    
    blocks = env.blocks
    targets = env.targets
    best_dir = 0
    best_locks = 0
    best_moves = 0
    
    for direction in [1, 2, 3, 4]:
        temp_blocks = [{'pos': b['pos'], 'locked': b['locked'], 'id': b['id']} for b in blocks]
        occupied = {b['pos']: True for b in temp_blocks}
        movable = [b for b in temp_blocks if not b['locked']]
        
        if direction == 1:
            dx, dy, key, rev = 0, -1, 1, False
        elif direction == 2:
            dx, dy, key, rev = 0, 1, 1, True
        elif direction == 3:
            dx, dy, key, rev = -1, 0, 0, False
        else:
            dx, dy, key, rev = 1, 0, 0, True
            
        movable_sorted = sorted(movable, key=lambda b: b['pos'][key], reverse=rev)
        new_positions = {}
        moved_count = 0
        
        for block in movable_sorted:
            start = block['pos']
            current = start
            while True:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < env.GRID_WIDTH and 0 <= nxt[1] < env.GRID_HEIGHT):
                    break
                if nxt in occupied:
                    break
                current = nxt
            new_positions[block['id']] = current
            if current != start:
                moved_count += 1
            occupied.pop(start)
            occupied[current] = True

        locks = 0
        for bid, pos in new_positions.items():
            if pos == targets[bid]['pos']:
                locks += 1
                
        if locks > best_locks or (locks == best_locks and moved_count > best_moves):
            best_locks = locks
            best_moves = moved_count
            best_dir = direction
            
    return [best_dir, 0, 0] if best_dir != 0 else [0, 0, 0]