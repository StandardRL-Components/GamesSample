def policy(env):
    # This policy for 2048 uses a heuristic that prioritizes moves that maximize immediate rewards
    # through tile merging while maintaining board organization. It simulates all possible moves,
    # evaluates them based on reward potential and board openness, and selects the best move.
    # Secondary actions are unused in this environment and set to 0.
    if env.game_over:
        return [0, 0, 0]
    
    board = env.board.tolist()
    n = len(board)
    best_score = -1
    best_dir = 1
    
    def rotate_ccw(grid):
        return [list(row) for row in zip(*grid)][::-1]
    
    def process_row(row):
        non_zero = [x for x in row if x != 0]
        new_row = []
        reward = 0
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_val = non_zero[i] * 2
                new_row.append(merged_val)
                reward += merged_val
                i += 2
            else:
                new_row.append(non_zero[i])
                i += 1
        new_row += [0] * (n - len(new_row))
        return new_row, reward
    
    def simulate_move(grid, direction):
        rotations = {1: 1, 2: 3, 3: 0, 4: 2}
        rot = rotations[direction]
        temp = [row[:] for row in grid]
        for _ in range(rot):
            temp = rotate_ccw(temp)
        new_temp = []
        total_reward = 0
        for row in temp:
            new_row, reward = process_row(row)
            new_temp.append(new_row)
            total_reward += reward
        for _ in range(4 - rot):
            new_temp = rotate_ccw(new_temp)
        return new_temp, total_reward
    
    for direction in [1, 2, 3, 4]:
        new_board, reward = simulate_move(board, direction)
        if new_board == board:
            continue
        empty_count = sum(row.count(0) for row in new_board)
        score = reward * 10 + empty_count
        if score > best_score:
            best_score = score
            best_dir = direction
    
    return [best_dir, 0, 0]