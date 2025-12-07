def policy(env):
    """
    Maximizes reward by maintaining high speed, avoiding obstacles, and staying centered.
    Prioritizes obstacle avoidance by steering towards clearest lane, accelerates when safe,
    and brakes only when necessary. Secondary actions (drift/fire) are unused in this environment.
    """
    # Early return if game over
    if env.game_over:
        return [0, 0, 0]
    
    # Constants
    danger_zone_top = 120
    danger_zone_bottom = 320
    road_left = env.road_edge_left
    road_right = env.road_edge_right
    lane_width = env.road_width / 3
    left_lane = (road_left, road_left + lane_width)
    center_lane = (road_left + lane_width, road_right - lane_width)
    right_lane = (road_right - lane_width, road_right)
    
    def check_lane_obstacle(lane_left, lane_right):
        for obs in env.obstacles:
            screen_y = obs['world_y'] - env.scroll_y
            if (screen_y <= danger_zone_bottom and 
                screen_y + obs['h'] >= danger_zone_top and
                not (obs['x'] + obs['w'] < lane_left or obs['x'] > lane_right)):
                return True
        return False
    
    # Check obstacles in each lane
    left_obstacle = check_lane_obstacle(*left_lane)
    center_obstacle = check_lane_obstacle(*center_lane)
    right_obstacle = check_lane_obstacle(*right_lane)
    
    # Determine current lane
    current_x = env.player_pos[0]
    if current_x < left_lane[1]:
        current_lane = 'left'
    elif current_x < center_lane[1]:
        current_lane = 'center'
    else:
        current_lane = 'right'
    
    # Edge avoidance
    if current_x < road_left + 20:
        return [4, 0, 0]  # Right
    if current_x > road_right - 20:
        return [3, 0, 0]  # Left
    
    # Obstacle avoidance
    current_has_obstacle = (
        (current_lane == 'left' and left_obstacle) or
        (current_lane == 'center' and center_obstacle) or
        (current_lane == 'right' and right_obstacle)
    )
    
    if current_has_obstacle:
        if not left_obstacle and current_x > road_left + 20:
            return [3, 0, 0]  # Left
        elif not right_obstacle and current_x < road_right - 20:
            return [4, 0, 0]  # Right
        else:
            return [2, 0, 0]  # Brake
    
    # Lane centering and acceleration
    if current_lane == 'left':
        return [4, 0, 0]  # Right toward center
    elif current_lane == 'right':
        return [3, 0, 0]  # Left toward center
    else:
        return [1, 0, 0]  # Accelerate in center lane