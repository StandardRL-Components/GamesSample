def policy(env):
    # Strategy: Choose moves that push the target towards the exit or position the player for future pushes.
    # For each direction, score based on potential distance reduction or proximity to ideal push position.
    # No-op is a fallback if all moves are detrimental due to move cost or impossibility.
    px, py = env.player_pos
    tx, ty = env.target_pos
    ex, ey = env.exit_pos
    grid = env.grid
    GRID_SIZE = env.GRID_SIZE
    dirs = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    best_score = -float('inf')
    best_a0 = 0
    for a0 in range(5):
        if a0 == 0:
            score = 0
        else:
            dx, dy = dirs[a0]
            nx, ny = px + dx, py + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                score = -10
            else:
                cell_content = grid[ny][nx]
                if cell_content is None:
                    dx_te = ex - tx
                    dy_te = ey - ty
                    if abs(dx_te) > abs(dy_te):
                        if dx_te > 0:
                            ideal_x = tx - 1
                            ideal_y = ty
                        else:
                            ideal_x = tx + 1
                            ideal_y = ty
                    else:
                        if dy_te > 0:
                            ideal_x = tx
                            ideal_y = ty - 1
                        else:
                            ideal_x = tx
                            ideal_y = ty + 1
                    dist = abs(nx - ideal_x) + abs(ny - ideal_y)
                    score = -dist
                else:
                    current_x, current_y = nx, ny
                    chain = []
                    while 0 <= current_x < GRID_SIZE and 0 <= current_y < GRID_SIZE and grid[current_y][current_x] is not None:
                        chain.append((current_x, current_y))
                        current_x += dx
                        current_y += dy
                    end_x, end_y = current_x, current_y
                    push_possible = 0 <= end_x < GRID_SIZE and 0 <= end_y < GRID_SIZE and grid[end_y][end_x] is None
                    target_in_chain = any(grid[cy][cx] == "target" for (cx, cy) in chain)
                    if push_possible and target_in_chain:
                        new_tx = tx + dx
                        new_ty = ty + dy
                        new_dist = abs(new_tx - ex) + abs(new_ty - ey)
                        old_dist = abs(tx - ex) + abs(ty - ey)
                        score = old_dist - new_dist
                    else:
                        score = -1
        if score > best_score:
            best_score = score
            best_a0 = a0
    return [best_a0, 0, 0]