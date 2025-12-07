def policy(env):
    # Strategy: Jump when the player is near the edge of the current platform and the next platform is within reachable horizontal distance.
    # This maximizes reward by ensuring successful jumps to new platforms (+1 reward) while avoiding falls from jumping too early or late.
    if env.on_ground and env.current_platform_index < len(env.platforms) - 1:
        current_platform = env.platforms[env.current_platform_index]
        next_platform = env.platforms[env.current_platform_index + 1]
        
        # Calculate horizontal distance to jump point (edge of current platform)
        player_right = env.player_pos[0] + env.PLAYER_SIZE
        platform_edge = current_platform.right
        
        # Jump when player is near edge and next platform is reachable
        edge_proximity = platform_edge - player_right
        horizontal_gap = next_platform.x - platform_edge
        
        if edge_proximity < 15 and horizontal_gap < 200:  # Empirical thresholds
            return [0, 1, 0]
    return [0, 0, 0]