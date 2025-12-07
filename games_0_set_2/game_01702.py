import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import os
import os
import pygame


# Set headless mode for Pygame
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
        self.font = pygame.font.Font(None, 24)

        # Define colors
        self.COLOR_BG = (50, 50, 50)
        self.COLOR_PLAYER = (255, 0, 0)
        self.COLOR_TEXT = (255, 255, 255)

        # Game state variables (initialized in reset)
        self.player = {}
        self.steps = 0
        self.score = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {
            'x': self.width / 2,
            'y': self.height / 2,
            'angle': 90,  # Start facing up
            'speed': 0,
            'width': 15,
            'height': 30
        }
        
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
        self._handle_input(movement, space_held, shift_held)
        self._update_player_position()
        self.steps += 1
        
        reward = self._calculate_reward()
        self.score += reward

        terminated = self._check_termination()
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Constants for car physics
        ACCELERATION = 0.2
        BRAKING = 0.5
        FRICTION = 0.05
        MAX_SPEED = 5.0
        TURN_SPEED = 3.0

        # Movement
        if movement == 1:  # Up
            self.player['speed'] = min(self.player['speed'] + ACCELERATION, MAX_SPEED)
        elif movement == 2:  # Down
            self.player['speed'] = max(self.player['speed'] - BRAKING, 0)
        
        # Turning (only when moving)
        if self.player['speed'] > 0.1:
            turn_direction = 0
            if movement == 3:  # Left
                turn_direction = 1
            elif movement == 4:  # Right
                turn_direction = -1
            self.player['angle'] += turn_direction * TURN_SPEED

        # Apply friction
        self.player['speed'] = max(self.player['speed'] - FRICTION, 0)
        self.player['angle'] %= 360

    def _update_player_position(self):
        angle_rad = math.radians(self.player['angle'])
        self.player['x'] += self.player['speed'] * math.cos(angle_rad)
        self.player['y'] -= self.player['speed'] * math.sin(angle_rad)

        # Boundary wrap-around
        if self.player['x'] < 0: self.player['x'] = self.width
        if self.player['x'] > self.width: self.player['x'] = 0
        if self.player['y'] < 0: self.player['y'] = self.height
        if self.player['y'] > self.height: self.player['y'] = 0

    def _calculate_reward(self):
        # Small reward for moving
        return 0.1 if self.player['speed'] > 1.0 else -0.01

    def _check_termination(self):
        # End after a fixed number of steps
        if self.steps >= 2000:
            self.game_over = True
        return self.game_over

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
        # Draw the player's car as a rotated rectangle
        car_w = self.player['width']
        car_h = self.player['height']
        
        # Create a car surface and rotate it
        car_surface = pygame.Surface((car_h, car_w), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, self.COLOR_PLAYER, (0, 0, car_h, car_w))
        rotated_surface = pygame.transform.rotate(car_surface, self.player['angle'])
        
        # Get the rect of the rotated surface and center it on the player's position
        rotated_rect = rotated_surface.get_rect(center=(self.player['x'], self.player['y']))
        
        self.screen.blit(rotated_surface, rotated_rect)

    def _render_ui(self):
        score_text = self.font.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        steps_text = self.font.render(f"Steps: {self.steps}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (10, 30))

    def close(self):
        pygame.quit()