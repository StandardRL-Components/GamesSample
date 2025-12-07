def policy(env):
    """
    Strategy: Maintain rider near screen center (y=200) by adjusting track slope based on vertical position error.
    Use steeper slopes for larger errors to prevent falls and maintain momentum toward the finish line.
    Default to flat track extension (a0=0) when within deadband to maximize horizontal progress.
    """
    rider_y = env.rider['pos'].y
    target_y = 200  # Screen center
    error = rider_y - target_y
    
    if error > 50:  # Too low: draw upward slope
        a0 = 1  # Up
        a1 = 1 if error > 100 else 0  # Steeper incline if very low
        a2 = 0
    elif error < -50:  # Too high: draw downward slope
        a0 = 2  # Down
        a1 = 0
        a2 = 1 if error < -100 else 0  # Steeper decline if very high
    else:  # Within acceptable range: extend flat track
        a0 = 0  # Flat right extension
        a1 = 0
        a2 = 0
        
    return [a0, a1, a2]