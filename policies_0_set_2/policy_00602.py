def policy(env):
    # Strategy: Maximize immediate line clears and minimize future holes by evaluating potential placements.
    # For each rotation and horizontal position, simulate the drop and score the resulting board based on:
    # lines cleared, aggregate height, holes, and bumpiness. Choose the best placement and return actions to achieve it.
    def evaluate_board(board, height, width):
        lines_cleared = 0
        for r in range(height):
            if all(board[r][c] != 0 for c in range(width)):
                lines_cleared += 1
        col_heights = [0] * width
        holes = 0
        for c in range(width):
            found_block = False
            for r in range(height):
                if board[r][c] != 0:
                    found_block = True
                    col_heights[c] = height - r
                    break
            if not found_block:
                col_heights[c] = 0
            for r in range(height):
                if board[r][c] == 0 and any(board[r2][c] != 0 for r2 in range(r)):
                    holes += 1
        aggregate_height = sum(col_heights)
        bumpiness = sum(abs(col_heights[i] - col_heights[i+1]) for i in range(width-1))
        return lines_cleared * 100 - aggregate_height * 0.5 - holes * 10 - bumpiness * 0.2

    if env.game_over:
        return [0, 0, 0]
    board = env.board
    current_piece = env.current_piece
    if current_piece is None:
        return [0, 0, 0]
    piece_type = current_piece['type']
    rotations = [env.TETROMINOES[piece_type]]
    for i in range(3):
        rotated = [list(row) for row in zip(*rotations[-1][::-1])]
        rotations.append(rotated)
    unique_rotations = []
    for rot in rotations:
        if rot not in unique_rotations:
            unique_rotations.append(rot)
    best_score = -10**9
    best_rotation = None
    best_x = None
    for rot in unique_rotations:
        width = len(rot[0])
        for x in range(env.BOARD_WIDTH - width + 1):
            y = 0
            while not env._check_collision(rot, (x, y+1)):
                y += 1
            temp_board = [list(row) for row in board]
            for r, row in enumerate(rot):
                for c, cell in enumerate(row):
                    if cell and y+r < env.BOARD_HEIGHT:
                        temp_board[y+r][x+c] = piece_type+1
            score = evaluate_board(temp_board, env.BOARD_HEIGHT, env.BOARD_WIDTH)
            if score > best_score:
                best_score = score
                best_rotation = rot
                best_x = x
    current_rotation = current_piece['shape']
    current_x = current_piece['x']
    if current_rotation != best_rotation:
        return [1, 0, 0]
    if current_x < best_x:
        return [4, 0, 0]
    elif current_x > best_x:
        return [3, 0, 0]
    return [0, 1, 0]