import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
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

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.COLOR_BG = (20, 20, 20)
        self.COLOR_TRACK = (100, 100, 100)
        self.COLOR_PLAYER = (200, 0, 50)
        self.COLOR_TEXT = (255, 255, 255)
        self.font = pygame.font.Font(None, 24)

        # Game parameters
        self.MAX_STEPS = 1000
        self.CAR_WIDTH = 20
        self.CAR_HEIGHT = 40
        self.MAX_SPEED = 6.0
        self.ACCELERATION = 0.2
        self.BRAKE_FORCE = 0.4
        self.FRICTION = 0.05
        self.TURN_SPEED = 0.05
        self.DRIFT_TURN_MULTIPLIER = 1.5

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_angle = None
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Player state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float64)
        self.player_vel = 0.0
        self.player_angle = -math.pi / 2  # Start facing up

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        self._update_player(movement, shift_held)
        
        self.steps += 1
        reward = self._calculate_reward()
        terminated = self._check_termination()
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _update_player(self, movement, shift_held):
        # 1. Handle player input
        turn_speed_multiplier = self.DRIFT_TURN_MULTIPLIER if shift_held else 1.0

        if movement == 1:  # Up
            self.player_vel += self.ACCELERATION
        elif movement == 2:  # Down
            self.player_vel -= self.BRAKE_FORCE
        elif movement == 3:  # Left
            if self.player_vel > 0.1: # Can only turn when moving
                self.player_angle -= self.TURN_SPEED * turn_speed_multiplier
        elif movement == 4:  # Right
            if self.player_vel > 0.1: # Can only turn when moving
                self.player_angle += self.TURN_SPEED * turn_speed_multiplier

        # 2. Apply physics
        # Clamp speed
        self.player_vel = np.clip(self.player_vel, 0, self.MAX_SPEED)
        # Apply friction
        self.player_vel *= (1 - self.FRICTION)
        if self.player_vel < 0: self.player_vel = 0

        # Update position
        dx = self.player_vel * math.cos(self.player_angle)
        dy = self.player_vel * math.sin(self.player_angle)
        self.player_pos += np.array([dx, dy])

        # 3. Handle boundaries (simple wrap around)
        self.player_pos[0] %= self.WIDTH
        self.player_pos[1] %= self.HEIGHT

    def _calculate_reward(self):
        # Reward for moving fast
        return self.player_vel / self.MAX_SPEED * 0.1

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw a simple track
        track_rect = pygame.Rect(50, 50, self.WIDTH - 100, self.HEIGHT - 100)
        pygame.draw.rect(self.screen, self.COLOR_TRACK, track_rect)
        pygame.draw.rect(self.screen, self.COLOR_BG, track_rect, 5)

        # Draw the player car
        car_surface = pygame.Surface((self.CAR_HEIGHT, self.CAR_WIDTH), pygame.SRCALPHA)
        car_surface.fill(self.COLOR_PLAYER)
        # Add a "front" indicator
        pygame.draw.rect(car_surface, (255, 255, 0), (self.CAR_HEIGHT * 0.7, 0, self.CAR_HEIGHT * 0.3, self.CAR_WIDTH))

        # Rotate the car surface and position it
        angle_deg = -math.degrees(self.player_angle) - 90
        rotated_car = pygame.transform.rotate(car_surface, angle_deg)
        new_rect = rotated_car.get_rect(center=tuple(self.player_pos))

        self.screen.blit(rotated_car, new_rect.topleft)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 30))

    def close(self):
        pygame.quit()