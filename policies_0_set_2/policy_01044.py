def policy(env):
    """
    Strategy: Maximize forward progress by drawing a continuous track slightly below the sled to catch it as it falls due to gravity.
    Always snap to the sled (shift held) and draw (space held) to extend the track. Move right 10 steps then down 3 steps in a cycle to create a descending track that supports the sled's motion.
    """
    if env.steps % 13 < 10:
        movement = 4  # Right
    else:
        movement = 2  # Down
    return [movement, 1, 1]  # Always draw and snap to sled