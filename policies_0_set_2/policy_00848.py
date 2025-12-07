def policy(env):
    """
    Strategy: Simulates each tilt direction to maximize immediate alignments of 5 (win) or 4 crystals.
    Prioritizes moves that create wins, then aligns 4, then avoids no-ops. Breaks ties by direction order.
    """
    def count_alignments(grid):
        top_color = {}
        for pos, stack in grid.items():
            if stack:
                top_color[pos] = stack[-1]['color_idx']
        fours, fives = 0, 0
        found = []
        for color in range(5):
            for dx, dy in [(1,0), (0,1), (1,1), (1,-1)]:
                for x in range(env.GRID_WIDTH):
                    for y in range(env.GRID_HEIGHT):
                        line = []
                        for k in range(5):
                            px, py = x + k*dx, y + k*dy
                            if (px, py) in top_color and top_color[(px, py)] == color:
                                line.append((px, py))
                            else:
                                break
                        if len(line) >= 4:
                            flat_found = [p for align in found for p in align]
                            if not any(p in flat_found for p in line):
                                if len(line) >= 5:
                                    fives += 1
                                else:
                                    fours += 1
                                found.append(line)
        return fours, fives

    def simulate_move(move):
        grid_copy = {}
        for pos, stack in env.grid.items():
            grid_copy[pos] = [{'id': c['id'], 'pos': c['pos'], 'color_idx': c['color_idx']} for c in stack]
        if move == 0:
            return grid_copy
        dx, dy = (0,-1) if move==1 else (0,1) if move==2 else (-1,0) if move==3 else (1,0)
        occupied = sorted(grid_copy.keys(), key=lambda p: -p[0]*dx - p[1]*dy)
        for x, y in occupied:
            if (x,y) not in grid_copy:
                continue
            stack = grid_copy.pop((x,y))
            cx, cy = x, y
            while True:
                nx, ny = cx+dx, cy+dy
                if not (0<=nx<env.GRID_WIDTH and 0<=ny<env.GRID_HEIGHT) or (nx,ny) in grid_copy:
                    break
                cx, cy = nx, ny
            grid_copy[(cx,cy)] = stack
            for c in stack:
                c['pos'] = (cx, cy)
        return grid_copy

    cur_fours, cur_fives = count_alignments(env.grid)
    if cur_fives > 0:
        return [0, 0, 0]

    best_score = -1
    best_move = 0
    for move in [1, 2, 3, 4]:
        new_grid = simulate_move(move)
        fours, fives = count_alignments(new_grid)
        score = 1000 * fives + 100 * fours
        if score > best_score:
            best_score = score
            best_move = move
    return [best_move, 0, 0]