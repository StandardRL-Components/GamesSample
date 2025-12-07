import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to drive, ←→ to turn and ↓ to brake. Hold shift to drift and press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Drift through corners, grab boosts, and use fire at your opponents."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Constants
        self.COLOR_BG = (50, 50, 50)  # Fix: Define color before use

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Initialize state variables. The original code called reset() here,
        # which is an anti-pattern, but we must fix the error within that structure.
        # The state is properly initialized by the reset() call below.
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # This call was in the original code and caused the error.
        # Now that COLOR_BG is defined, it will work correctly.
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        self.steps += 1
        reward = self._calculate_reward()
        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        # Transpose from Pygame's (W, H, C) to Gym's (H, W, C)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    # --- Placeholder methods to make the class runnable ---

    def _render_game(self):
        """Placeholder for rendering the main game scene."""
        pass

    def _render_ui(self):
        """Placeholder for rendering UI elements."""
        pass

    def _calculate_reward(self):
        """Placeholder for reward calculation logic."""
        return 0.0

    def _check_termination(self):
        """Placeholder for termination condition logic."""
        # Example: end episode after a fixed number of steps
        if self.steps >= 1000:
            return True
        return self.game_over

    def render(self):
        """Standard render method for Gymnasium."""
        return self._get_observation()

    def close(self):
        """Standard close method for Gymnasium."""
        pygame.quit()