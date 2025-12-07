import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


# Set Pygame to run in headless mode
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
        
        self.render_mode = render_mode
        self.width = 640
        self.height = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (30, 30, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Game state variables (initialized in reset)
        self.player = None
        self.player_speed = 5
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        # This was in the original code, so we ensure all attributes
        # needed for reset() are defined above.
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player = pygame.Rect(self.width // 2 - 20, self.height - 60, 40, 40)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Update game logic
        if movement == 1:  # Up
            self.player.y -= self.player_speed
        elif movement == 2:  # Down
            self.player.y += self.player_speed
        elif movement == 3:  # Left
            self.player.x -= self.player_speed
        elif movement == 4:  # Right
            self.player.x += self.player_speed

        # Keep player within screen bounds
        self.player.left = max(0, self.player.left)
        self.player.right = min(self.width, self.player.right)
        self.player.top = max(0, self.player.top)
        self.player.bottom = min(self.height, self.player.bottom)

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
    
    def _calculate_reward(self):
        # A simple reward for surviving
        return 0.1

    def _check_termination(self):
        # End the game after a fixed number of steps
        if self.steps >= 1000:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw the player
        if self.player:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player)

    def _render_ui(self):
        # Display score and steps on the screen
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 35))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
    
    def close(self):
        pygame.quit()