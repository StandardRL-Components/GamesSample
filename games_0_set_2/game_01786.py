import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


# Set up headless Pygame
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
        self.width, self.height = 640, 400

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

        # Colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 40
        self.MAX_SPEED = 5.0
        self.ACCELERATION = 0.2
        self.BRAKE_POWER = 0.3
        self.FRICTION = 0.05
        self.TURN_SPEED = 0.05
        self.MAX_STEPS = 2000

        # Initialize state variables
        self.player_pos = None
        self.player_angle = None
        self.player_speed = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        # This will be initialized in reset()
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        # Initialize all game state
        self.player_pos = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float64)
        self.player_angle = 0.0  # Angle in radians
        self.player_speed = 0.0
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
            
        self._update_game_state(action)
        
        reward = self._calculate_reward()
        self.score += reward
        
        self.steps += 1
        terminated = self._check_termination()
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_state(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held = action[1] == 1  # Boolean (unused)
        # shift_held = action[2] == 1  # Boolean (unused)

        # Turning
        if movement == 3:  # Left
            self.player_angle -= self.TURN_SPEED * (abs(self.player_speed) / self.MAX_SPEED)
        if movement == 4:  # Right
            self.player_angle += self.TURN_SPEED * (abs(self.player_speed) / self.MAX_SPEED)

        # Acceleration/Braking
        if movement == 1:  # Up
            self.player_speed += self.ACCELERATION
        elif movement == 2:  # Down
            self.player_speed -= self.BRAKE_POWER
        
        # Apply friction
        if self.player_speed > 0:
            self.player_speed -= self.FRICTION
            if self.player_speed < 0: self.player_speed = 0
        elif self.player_speed < 0:
            self.player_speed += self.FRICTION
            if self.player_speed > 0: self.player_speed = 0

        # Clamp speed
        self.player_speed = np.clip(self.player_speed, -self.MAX_SPEED / 2, self.MAX_SPEED)

        # Update position
        dx = self.player_speed * math.sin(self.player_angle)
        dy = -self.player_speed * math.cos(self.player_angle) # Pygame y-axis is inverted
        self.player_pos += np.array([dx, dy])

        # Keep player on screen (wrap around)
        self.player_pos[0] %= self.width
        self.player_pos[1] %= self.height

    def _calculate_reward(self):
        # Small reward for surviving
        return 0.01

    def _check_termination(self):
        # End episode after a fixed number of steps
        return self.steps >= self.MAX_STEPS

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw the player's car
        car_rect = pygame.Rect(0, 0, self.CAR_WIDTH, self.CAR_HEIGHT)
        car_rect.center = (0, 0)
        
        corners = [car_rect.topleft, car_rect.topright, car_rect.bottomright, car_rect.bottomleft]
        
        # Rotate and translate corners
        rotated_corners = []
        for x, y in corners:
            x_rot = x * math.cos(self.player_angle) - y * math.sin(self.player_angle)
            y_rot = x * math.sin(self.player_angle) + y * math.cos(self.player_angle)
            rotated_corners.append((x_rot + self.player_pos[0], y_rot + self.player_pos[1]))
            
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, rotated_corners)

    def _render_ui(self):
        # Render score
        score_text = self.font.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Render steps
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.width - steps_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }
        
    def close(self):
        pygame.quit()