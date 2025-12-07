def policy(env):
    # Strategy: Always jump to the furthest platform to the right to maximize progress and risky jump rewards.
    # This balances risk and reward by prioritizing long jumps (higher bonus) while moving toward higher-numbered platforms.
    # Avoids falling by ensuring jumps are within max radius (env handles validation).
    return [4, 1, 0]  # Right + Space (furthest jump)