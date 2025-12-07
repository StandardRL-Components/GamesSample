def policy(env):
    # This policy simulates each push direction to minimize total Manhattan distance of unsolved blocks to their goals.
    # It prioritizes moves that reduce overall distance and avoids no-ops by checking if blocks actually move.
    def simulate_push(blocks, solved_blocks, walls, goals_dict, dx, dy):
        occupied = set(walls)
        current_unsolved = {block['id']: block['pos'] for block in blocks if block['id'] not in solved_blocks}
        for block in blocks:
            if block['id'] in solved_blocks:
                occupied.add(block['pos'])
        sorted_blocks = sorted(current_unsolved.items(), key=lambda item: item[1][0] * -dx + item[1][1] * -dy)
        new_positions = {}
        moved = False
        for block_id, pos in sorted_blocks:
            x, y = pos
            new_pos = (x + dx, y + dy)
            if new_pos in occupied:
                new_positions[block_id] = pos
                occupied.add(pos)
            else:
                new_positions[block_id] = new_pos
                occupied.add(new_pos)
                if pos != new_pos:
                    moved = True
        return new_positions, moved

    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    best_action = 1
    best_score = float('inf')
    any_move = False
    for idx, (dx, dy) in enumerate(directions):
        new_positions, moved = simulate_push(env.blocks, env.solved_blocks, env.walls, env.goals_dict, dx, dy)
        if not moved:
            continue
        any_move = True
        total_dist = 0
        for block_id, pos in new_positions.items():
            goal_pos = env.goals_dict[block_id]['pos']
            total_dist += abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
        if total_dist < best_score:
            best_score = total_dist
            best_action = idx + 1
    if not any_move:
        best_action = 1
    return [best_action, 1, 0]