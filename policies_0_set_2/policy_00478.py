def policy(env):
    # This policy minimizes total Manhattan distance to goals by simulating each push direction.
    # It avoids no-op moves when possible and breaks ties consistently (up > down > left > right).
    if env.game_over:
        return [0, 0, 0]
    
    current_positions = [tuple(block['pos']) for block in env.blocks]
    current_distance = sum(
        abs(block['pos'][0] - env.goals[block['color_idx']]['pos'][0]) + 
        abs(block['pos'][1] - env.goals[block['color_idx']]['pos'][1])
        for block in env.blocks
    )
    
    def simulate_push(positions, direction):
        if direction == 1:  # Up
            sort_order = sorted(range(len(positions)), key=lambda k: positions[k][1])
            dx, dy = 0, -1
        elif direction == 2:  # Down
            sort_order = sorted(range(len(positions)), key=lambda k: positions[k][1], reverse=True)
            dx, dy = 0, 1
        elif direction == 3:  # Left
            sort_order = sorted(range(len(positions)), key=lambda k: positions[k][0])
            dx, dy = -1, 0
        else:  # Right
            sort_order = sorted(range(len(positions)), key=lambda k: positions[k][0], reverse=True)
            dx, dy = 1, 0

        new_positions = list(positions)
        for i in sort_order:
            target = (new_positions[i][0] + dx, new_positions[i][1] + dy)
            if (0 <= target[0] < env.GRID_COLS and 0 <= target[1] < env.GRID_ROWS and 
                target not in new_positions):
                new_positions[i] = target
        return new_positions

    best_dir = 0
    best_dist = current_distance
    for direction in [1, 2, 3, 4]:
        new_positions = simulate_push(current_positions, direction)
        if new_positions == current_positions:
            continue
        dist = sum(
            abs(new_positions[i][0] - env.goals[env.blocks[i]['color_idx']]['pos'][0]) + 
            abs(new_positions[i][1] - env.goals[env.blocks[i]['color_idx']]['pos'][1])
            for i in range(len(env.blocks))
        )
        if dist < best_dist:
            best_dist = dist
            best_dir = direction

    return [best_dir, 0, 0] if best_dir != 0 else [0, 0, 0]