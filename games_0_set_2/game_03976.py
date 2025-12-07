import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import math
import os
import pygame


# Set Pygame to run in a headless mode
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

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Define colors
        self.COLOR_BG = (50, 50, 50)  # Asphalt gray
        self.COLOR_PLAYER = (255, 0, 0)  # Red
        self.COLOR_UI_TEXT = (255, 255, 255) # White

        # Font for UI
        self.font = pygame.font.Font(None, 36)
        
        # Player properties
        self.player_pos = None
        self.player_angle = 0.0
        self.player_speed = 0.0
        self.player_turn_speed = 5.0
        self.player_acceleration = 0.2
        self.player_deceleration = 0.1
        self.player_max_speed = 5.0

        # Initialize state variables
        # The traceback shows this is called from init, so we keep it.
        # In general, it's better to let the user call reset() after creation.
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Reset player state
        self.player_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float64)
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
        if movement == 1:  # Up
            self.player_speed = min(self.player_max_speed, self.player_speed + self.player_acceleration)
        elif movement == 2:  # Down
            self.player_speed = max(-self.player_max_speed / 2, self.player_speed - self.player_acceleration * 2)
        
        if self.player_speed != 0:
            if movement == 3:  # Left
                self.player_angle += self.player_turn_speed
            elif movement == 4:  # Right
                self.player_angle -= self.player_turn_speed
        
        # Apply natural deceleration
        if self.player_speed > 0:
            self.player_speed = max(0, self.player_speed - self.player_deceleration)
        elif self.player_speed < 0:
            self.player_speed = min(0, self.player_speed + self.player_deceleration)

        # Update player position
        angle_rad = math.radians(self.player_angle)
        self.player_pos[0] += self.player_speed * math.sin(angle_rad)
        self.player_pos[1] -= self.player_speed * math.cos(angle_rad)

        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.screen_width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.screen_height)

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
        # Draw the player as a circle
        player_pos_int = self.player_pos.astype(int)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_pos_int, 10)
        
        # Draw a line to indicate direction
        angle_rad = math.radians(self.player_angle)
        end_pos_x = player_pos_int[0] + 15 * math.sin(angle_rad)
        end_pos_y = player_pos_int[1] - 15 * math.cos(angle_rad)
        pygame.draw.line(self.screen, self.COLOR_UI_TEXT, player_pos_int, (end_pos_x, end_pos_y), 2)


    def _render_ui(self):
        # Render score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render steps
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def render(self):
        # The only supported render mode is 'rgb_array' which is handled by _get_observation
        return self._get_observation()

    def close(self):
        pygame.quit()