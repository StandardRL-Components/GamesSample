def policy(env):
    """
    This policy maximizes reward by prioritizing rightward movement to keep up with auto-scrolling,
    jumping over pits and platform gaps when detected within a safe distance (50 pixels), and
    maintaining momentum to collect movement rewards and avoid penalties.
    """
    if not env.on_ground:
        return [4, 0, 0]  # Always move right while airborne to maintain progress

    player_world_x = env.world_scroll_x + env.player_pos[0]
    player_feet_y = env.player_pos[1] + env.PLAYER_SIZE[1]

    # Check for imminent pits within 50 pixels
    for pit in env.pits:
        if pit.x > player_world_x and pit.x - player_world_x < 50:
            return [1, 0, 0]  # Jump over pit

    # Check if current platform ends within 50 pixels
    for plat in env.platforms:
        plat_world_x = plat.x
        plat_world_right = plat.right
        plat_top = plat.y
        if (player_feet_y >= plat_top and player_feet_y <= plat_top + 5 and
            player_world_x >= plat_world_x and player_world_x <= plat_world_right and
            plat_world_right - player_world_x < 50):
            return [1, 0, 0]  # Jump before platform edge

    return [4, 0, 0]  # Default: move right to maximize scrolling rewards