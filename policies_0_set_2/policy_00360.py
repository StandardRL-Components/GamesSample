def policy(env):
    """
    Maximizes immediate reward by selecting the move that yields the highest merge score and number of merges.
    Prefers moves that change the board to avoid penalties, with tie-breaking favoring up, left, right, then down.
    """
    if env.game_over:
        return [0, 0, 0]
    
    moves = [1, 3, 4, 2]  # up, left, right, down
    best_move = None
    best_score = -1
    
    for move in moves:
        rot_count = {1: 1, 3: 0, 4: 2, 2: 3}[move]
        rotated_board = np.rot90(env.board, rot_count)
        new_rotated_board, merge_score, merges_made = env._process_board(rotated_board)
        new_board = np.rot90(new_rotated_board, -rot_count)
        if not np.array_equal(env.board, new_board):
            score = merge_score + merges_made
            if best_move is None or score > best_score:
                best_score = score
                best_move = move
                
    if best_move is not None:
        return [best_move, 0, 0]
        
    return [1, 0, 0]