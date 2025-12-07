def policy(env):
    # Strategy: Jump when obstacles are approaching within optimal timing window (150-250 pixels ahead)
    # to clear them at peak jump height. Only jump when grounded to maximize jump effectiveness.
    player_right_edge = 120  # Player x=100 + size=20
    lookahead_min, lookahead_max = 150, 250
    for obstacle in env.obstacles:
        obstacle_screen_x = obstacle.x - env.world_scroll_x
        if lookahead_min <= obstacle_screen_x <= lookahead_max and env.on_ground:
            return [0, 1, 0]
    return [0, 0, 0]