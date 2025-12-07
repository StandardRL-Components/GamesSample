import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import math
import os
import pygame


# Set up Pygame to run in a headless environment
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

        # Screen dimensions
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
        self.font = pygame.font.Font(None, 36)

        # Game constants
        self.COLOR_BG = (50, 50, 50)  # Dark gray background
        self.COLOR_PLAYER = (255, 0, 0) # Red for the player
        self.COLOR_TEXT = (255, 255, 255) # White for UI text

        # Player state
        self.player_pos = np.array([0.0, 0.0])
        self.player_angle = 0.0
        self.player_speed = 0.0

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Reset player to center
        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=float)
        self.player_angle = 0.0
        self.player_speed = 0.0

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic based on action
        # 0: none, 1: up, 2: down, 3: left, 4: right
        turn_speed = 5.0
        acceleration = 0.5
        braking = 0.8
        max_speed = 10.0

        if movement == 1: # Up
            self.player_speed = min(self.player_speed + acceleration, max_speed)
        elif movement == 2: # Down
            self.player_speed = max(self.player_speed - braking, 0)
        
        if self.player_speed > 0:
            if movement == 3: # Left
                self.player_angle += turn_speed
            elif movement == 4: # Right
                self.player_angle -= turn_speed
        
        # Update player position
        self.player_pos[0] += self.player_speed * math.cos(math.radians(self.player_angle))
        self.player_pos[1] -= self.player_speed * math.sin(math.radians(self.player_angle)) # Pygame y-axis is inverted
        
        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.height)


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
        # Placeholder reward logic
        return 0.0

    def _check_termination(self):
        # End episode after a fixed number of steps
        return self.steps >= 1000

    def _render_game(self):
        # Draw a simple representation of the player (a rotated rectangle)
        player_size = (30, 15)
        player_rect = pygame.Rect((0, 0), player_size)
        player_rect.center = self.player_pos
        
        # Create a surface for the player and rotate it
        player_surf = pygame.Surface(player_size, pygame.SRCALPHA)
        player_surf.fill(self.COLOR_PLAYER)
        rotated_surf = pygame.transform.rotate(player_surf, self.player_angle)
        rotated_rect = rotated_surf.get_rect(center=player_rect.center)
        
        self.screen.blit(rotated_surf, rotated_rect)


    def _render_ui(self):
        # Render score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game()

        # Render UI overlay
        self._render_ui()

        # Convert to numpy array (EXACT format required)
        # Pygame's surfarray is typically (width, height, channels)
        arr = pygame.surfarray.array3d(self.screen)
        # Gymnasium expects (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()