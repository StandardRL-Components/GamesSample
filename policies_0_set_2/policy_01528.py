def policy(env):
    """
    Strategy: Jump on the beat to maximize rewards and avoid obstacles. 
    The beat indicator pulsates at 120 BPM (every 500ms). We sample the center 
    indicator region to detect beat timing, and jump when the intensity peaks 
    (indicating the beat window). This maximizes combo rewards and avoids penalties.
    """
    obs = env._get_observation()
    # Sample beat indicator region (center of screen)
    center_x, center_y = 320, 100  # WIDTH//2, HEIGHT//4
    max_intensity = 0
    for dx in (-2, 0, 2):
        for dy in (-2, 0, 2):
            r, g, b = obs[center_y + dy, center_x + dx]
            intensity = max(r, g, b)
            if intensity > max_intensity:
                max_intensity = intensity
    # Jump if beat indicator is bright (peak intensity) and player is likely grounded
    jump = 1 if max_intensity > 200 else 0
    return [0, jump, 0]