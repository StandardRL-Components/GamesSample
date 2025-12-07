def policy(env):
    """
    Maximizes reward by moving right towards the goal while maintaining risky proximity to obstacles.
    Prioritizes rightward movement with boost when safe, adjusts vertically to stay near obstacles
    without colliding, and uses a simple scoring system to evaluate candidate moves.
    """
    def hypot(x, y):
        return (x*x + y*y) ** 0.5

    def point_dist(p1, p2):
        return hypot(p1[0]-p2[0], p1[1]-p2[1])

    def dist_to_nearest_obstacle(rect):
        if not env.obstacles:
            return float('inf')
        min_dist = float('inf')
        center = rect.center
        for obs in env.obstacles:
            closest_x = max(obs.left, min(center[0], obs.right))
            closest_y = max(obs.top, min(center[1], obs.bottom))
            dist = hypot(center[0] - closest_x, center[1] - closest_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def in_bounds(rect):
        return (rect.left >= 0 and rect.right <= env.WIDTH and
                rect.top >= 0 and rect.bottom <= env.HEIGHT)

    if env.game_over:
        return [0, 0, 0]

    current_rect = env.player_rect
    finish_center = env.finish_rect.center
    current_center = current_rect.center
    current_dist_to_goal = point_dist(current_center, finish_center)

    best_score = -float('inf')
    best_action = [0, 0, 0]

    movements = [
        (4, 1, 0),   # right
        (1, 0, -1),  # up
        (2, 0, 1),   # down
        (0, 0, 0),   # none
        (3, -1, 0)   # left
    ]

    for move in movements:
        action_index, dx, dy = move
        for boost in [0, 1]:
            move_speed = 2 if boost else 1
            new_rect = current_rect.copy()
            new_rect.move_ip(dx * move_speed, dy * move_speed)

            if not in_bounds(new_rect):
                continue

            if any(new_rect.colliderect(obs) for obs in env.obstacles):
                continue

            new_center = new_rect.center
            if new_rect.colliderect(env.finish_rect):
                score = 100.0
            else:
                new_dist_to_goal = point_dist(new_center, finish_center)
                progress = current_dist_to_goal - new_dist_to_goal
                dist_obstacle = dist_to_nearest_obstacle(new_rect)
                is_moving = (dx != 0 or dy != 0)
                risky_bonus = 5.0 if (is_moving and dist_obstacle < env.RISKY_MOVE_DISTANCE) else -0.5
                score = progress + risky_bonus

            if score > best_score:
                best_score = score
                best_action = [action_index, boost, 0]

    return best_action