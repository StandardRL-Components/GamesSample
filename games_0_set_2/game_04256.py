import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
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

        # Colors
        self.COLOR_BG = (100, 100, 100)
        self.COLOR_PLAYER = (50, 150, 250)
        self.COLOR_TEXT = (255, 255, 255)

        # Game constants
        self.MAX_STEPS = 1000
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 40
        self.MAX_SPEED = 8.0
        self.ACCELERATION = 0.2
        self.BRAKE_POWER = 0.4
        self.FRICTION = 0.05
        self.TURN_SPEED = 0.05
        self.DRIFT_TURN_MULTIPLIER = 1.5

        # Initialize state variables
        self.player_pos = None
        self.player_speed = None
        self.player_angle = None
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.player_pos = np.array([self.width / 2, self.height / 2], dtype=np.float64)
        self.player_speed = 0.0
        self.player_angle = -math.pi / 2  # Start facing up
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
        space_held = action[1] == 1  # Boolean (weapon)
        shift_held = action[2] == 1  # Boolean (drift)

        # Update game logic
        self._update_player(movement, shift_held)
        self.steps += 1

        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        self.game_over = terminated

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_player(self, movement, shift_held):
        # Handle acceleration and braking
        if movement == 1:  # Up
            self.player_speed += self.ACCELERATION
        elif movement == 2:  # Down
            self.player_speed -= self.BRAKE_POWER
        
        self.player_speed = np.clip(self.player_speed, -self.MAX_SPEED/2, self.MAX_SPEED)

        # Apply friction
        if self.player_speed > 0:
            self.player_speed -= self.FRICTION
        elif self.player_speed < 0:
            self.player_speed += self.FRICTION
        if abs(self.player_speed) < self.FRICTION:
            self.player_speed = 0

        # Handle turning, only when moving
        if self.player_speed != 0:
            turn_multiplier = self.DRIFT_TURN_MULTIPLIER if shift_held else 1.0
            if movement == 3:  # Left
                self.player_angle -= self.TURN_SPEED * turn_multiplier
            elif movement == 4:  # Right
                self.player_angle += self.TURN_SPEED * turn_multiplier

        # Update position
        dx = self.player_speed * math.cos(self.player_angle)
        dy = self.player_speed * math.sin(self.player_angle)
        self.player_pos += np.array([dx, dy])

        # Keep player on screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.height)

    def _calculate_reward(self):
        # Reward for moving forward
        return self.player_speed * 0.1 if self.player_speed > 0 else 0

    def _check_termination(self):
        return self.steps >= self.MAX_STEPS

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

    def _render_game(self):
        # Draw player car
        if self.player_pos is not None:
            car_rect = pygame.Rect(0, 0, self.CAR_WIDTH, self.CAR_HEIGHT)
            car_rect.center = (0, 0)
            
            # Define corners relative to center
            corners = [car_rect.topleft, car_rect.topright, car_rect.bottomright, car_rect.bottomleft]
            
            # Rotate corners
            rotated_corners = []
            for x, y in corners:
                rotated_x = x * math.cos(-self.player_angle) - y * math.sin(-self.player_angle)
                rotated_y = x * math.sin(-self.player_angle) + y * math.cos(-self.player_angle)
                rotated_corners.append((rotated_x + self.player_pos[0], rotated_y + self.player_pos[1]))
            
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, rotated_corners)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()