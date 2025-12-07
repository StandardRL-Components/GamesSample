def policy(env):
    # Strategy: Compute Manhattan distance from each block to its target, then vote on the direction
    # that minimizes the total distance. Prioritize directions that move blocks toward targets.
    # This greedy approach efficiently reduces distances and achieves high reward by minimizing moves.
    votes = {1: 0, 2: 0, 3: 0, 4: 0}  # up, down, left, right
    for block in env.blocks:
        target = next(t for t in env.targets if t['block_id'] == block['id'])
        bx, by = block['rect'].x // env.CELL_SIZE, block['rect'].y // env.CELL_SIZE
        tx, ty = target['rect'].x // env.CELL_SIZE, target['rect'].y // env.CELL_SIZE
        dx, dy = tx - bx, ty - by
        if dx != 0:
            votes[4 if dx > 0 else 3] += abs(dx)
        if dy != 0:
            votes[2 if dy > 0 else 1] += abs(dy)
    best_dir = max(votes, key=votes.get) if max(votes.values()) > 0 else 0
    return [best_dir, 0, 0]