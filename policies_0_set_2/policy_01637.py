def policy(env):
    # Strategy: Prioritize moving balls to their matching slots to maximize immediate rewards and avoid penalties.
    # If a ball is selected, move it to its target slot. Otherwise, select the ball with the smallest combined distance (cursor-to-ball + ball-to-slot).
    # Avoid invalid placements by checking for blocking balls and deselecting when necessary.
    if env.selected_ball_idx is not None:
        selected_ball = env.balls[env.selected_ball_idx]
        target_slot = next((s for s in env.slots if s['color'] == selected_ball['color'] and not s['filled']), None)
        if target_slot is None:
            return [0, 0, 1]  # Deselect if no valid slot
        target_pos = target_slot['pos']
        blocking_ball = next((b for b in env.balls if b['pos'] == target_pos and b != selected_ball), None)
        if blocking_ball is not None:
            return [0, 0, 1]  # Deselect if slot is blocked
        if env.cursor_pos == target_pos:
            return [0, 1, 0]  # Place if at target
        dx = target_pos[0] - env.cursor_pos[0]
        dy = target_pos[1] - env.cursor_pos[1]
        if abs(dx) > abs(dy):
            move = 4 if dx > 0 else 3
        else:
            move = 2 if dy > 0 else 1
        return [move, 0, 0]
    else:
        candidate_balls = []
        for ball in env.balls:
            if ball['pos'][0] == -1:
                continue
            slot = next((s for s in env.slots if s['color'] == ball['color'] and not s['filled']), None)
            if slot is None:
                continue
            candidate_balls.append((ball, slot))
        if not candidate_balls:
            return [0, 0, 0]
        best_ball, best_slot = min(candidate_balls, key=lambda x: 
            abs(env.cursor_pos[0] - x[0]['pos'][0]) + abs(env.cursor_pos[1] - x[0]['pos'][1]) + 
            abs(x[0]['pos'][0] - x[1]['pos'][0]) + abs(x[0]['pos'][1] - x[1]['pos'][1]))
        target_pos = best_ball['pos']
        if env.cursor_pos == target_pos:
            return [0, 1, 0]  # Select if at ball
        dx = target_pos[0] - env.cursor_pos[0]
        dy = target_pos[1] - env.cursor_pos[1]
        if abs(dx) > abs(dy):
            move = 4 if dx > 0 else 3
        else:
            move = 2 if dy > 0 else 1
        return [move, 0, 0]