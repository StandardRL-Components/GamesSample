def policy(env):
    # Always accelerate to maximize speed-based reward (0.1 when speed > 1.0 vs -0.01 otherwise)
    return [1, 0, 0]