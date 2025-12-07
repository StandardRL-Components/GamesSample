def policy(env):
    # Strategy: Always hard drop the current piece to maximize line clears per step and minimize penalties from slow falling.
    # Avoid unnecessary moves or holds to prevent cooldown delays and focus on rapid placement.
    return [0, 1, 0]