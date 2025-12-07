def policy(env):
    # This policy simulates each push direction to maximize blocks on target and minimize distance to targets, avoiding no-op moves.
    if env.game_over or all(block.on_target for block in env.blocks):
        return [0, 0, 0]
    
    target_dict = {target.color: (target.grid_x, target.grid_y) for target in env.targets}
    current_positions = [(block.grid_x, block.grid_y) for block in env.blocks]
    best_score = -10**9
    best_direction = 0
    
    def simulate_push(direction, positions):
        n = len(positions)
        new_positions = positions[:]
        occupied = set(new_positions)
        if direction == 1:
            dx, dy = 0, -1
            indices = sorted(range(n), key=lambda i: new_positions[i][1])
        elif direction == 2:
            dx, dy = 0, 1
            indices = sorted(range(n), key=lambda i: new_positions[i][1], reverse=True)
        elif direction == 3:
            dx, dy = -1, 0
            indices = sorted(range(n), key=lambda i: new_positions[i][0])
        else:
            dx, dy = 1, 0
            indices = sorted(range(n), key=lambda i: new_positions[i][0], reverse=True)
        
        for i in indices:
            x, y = new_positions[i]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < env.GRID_COLS and 0 <= ny < env.GRID_ROWS):
                continue
            if (nx, ny) in occupied:
                continue
            occupied.discard((x, y))
            new_positions[i] = (nx, ny)
            occupied.add((nx, ny))
        return new_positions

    for d in [1, 2, 3, 4]:
        new_positions = simulate_push(d, current_positions)
        if new_positions == current_positions:
            score = -1000000
        else:
            score = 0
            for i, block in enumerate(env.blocks):
                tx, ty = target_dict[block.color]
                nx, ny = new_positions[i]
                if (nx, ny) == (tx, ty):
                    score += 1000
                else:
                    dist = abs(nx - tx) + abs(ny - ty)
                    score -= dist
        if score > best_score:
            best_score = score
            best_direction = d

    return [best_direction, 0, 0]