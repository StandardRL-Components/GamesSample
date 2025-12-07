import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import math
import os
import pygame


# Set up Pygame to run headlessly
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
        
        self.screen_width = 640
        self.screen_height = 400

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Etc...
        # Define colors
        self.COLOR_BG = (100, 120, 100)  # Greenish-gray for track/grass
        self.COLOR_CAR = (255, 0, 0)     # Red car
        self.COLOR_TEXT = (255, 255, 255) # White text

        # Game constants
        self.max_speed = 5.0
        self.acceleration = 0.2
        self.brake_power = 0.5
        self.turn_speed = 5.0
        self.friction = 0.05
        self.max_steps = 1000
        
        # Initialize state variables
        # These will be properly set in reset(), but need to exist before it's called.
        self.player_pos = np.array([0.0, 0.0])
        self.player_angle = 0.0
        self.player_speed = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initialize state by calling reset, as in the original code
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.player_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)
        self.player_angle = 0.0
        self.player_speed = 0.0
        
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
        
        # --- Player Controls ---
        # Turning (only when moving)
        if self.player_speed > 0.1:
            if movement == 3:  # Left
                self.player_angle += self.turn_speed
            if movement == 4:  # Right
                self.player_angle -= self.turn_speed
        
        # Acceleration/Braking
        if movement == 1:  # Up
            self.player_speed += self.acceleration
        elif movement == 2: # Down
            self.player_speed -= self.brake_power
        
        # Clamp speed
        self.player_speed = np.clip(self.player_speed, 0, self.max_speed)
        
        # Apply friction
        if self.player_speed > 0:
            self.player_speed -= self.friction
        if self.player_speed < 0:
            self.player_speed = 0

        # --- Update Position ---
        angle_rad = math.radians(self.player_angle)
        self.player_pos[0] += self.player_speed * math.sin(angle_rad)
        self.player_pos[1] -= self.player_speed * math.cos(angle_rad) # Pygame Y is inverted

        # --- Screen Boundaries (Wrap around) ---
        self.player_pos[0] %= self.screen_width
        self.player_pos[1] %= self.screen_height

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
        # Small reward for moving forward
        return 0.1 if self.player_speed > 1.0 else -0.01

    def _check_termination(self):
        # End episode after a fixed number of steps
        return self.steps >= self.max_steps

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
        # Draw the player's car as a rotated triangle
        car_size = 10
        angle_rad = math.radians(self.player_angle)
        
        points_rel = [
            (0, -car_size * 1.5),
            (-car_size, car_size),
            (car_size, car_size)
        ]

        def rotate(p, angle):
            x, y = p
            return (
                x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle)
            )

        rotated_points = [rotate(p, -angle_rad) for p in points_rel]
        
        car_points = [
            (self.player_pos[0] + p[0], self.player_pos[1] + p[1]) for p in rotated_points
        ]
        
        pygame.draw.polygon(self.screen, self.COLOR_CAR, car_points)

    def _render_ui(self):
        # Render score and steps
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(steps_text, (10, 40))

    def close(self):
        pygame.quit()