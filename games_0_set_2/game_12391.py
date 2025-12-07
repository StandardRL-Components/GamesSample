import gymnasium as gym
import os
import pygame
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    GameEnv: A Gymnasium environment where the player grows a line to a target length
    while navigating a field of obstacles.

    - The player controls the direction of line growth.
    - Colliding with an obstacle shrinks the line.
    - The goal is to reach a length of 200.
    - The game ends if the line length reaches 0 or 200, or if max steps are exceeded.
    - The visual style is minimalist and geometric, with a focus on clarity and feedback.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Grow your line to the target length while avoiding obstacles. "
        "Colliding with an obstacle will shrink your line."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to control the direction of the line's growth."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000

        # Game parameters
        self.INITIAL_LINE_LENGTH = 50.0
        self.TARGET_LINE_LENGTH = 200.0
        self.GROWTH_PER_STEP = 5.0
        self.SHRINK_ON_COLLISION = 25.0
        self.NUM_OBSTACLES = 15
        self.OBSTACLE_RADIUS = 10
        self.OBSTACLE_MIN_DIST = 10

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_OBSTACLE = (220, 60, 60)
        self.COLOR_OBSTACLE_OUTLINE = (150, 40, 40)
        self.COLOR_LINE_HEAD = pygame.Color(100, 255, 100)
        self.COLOR_LINE_TAIL = pygame.Color(30, 120, 30)
        self.COLOR_TEXT = (240, 240, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.line_length = 0.0
        self.line_segments = deque()
        self.obstacles = []

        # Initialize state for the first time to define all attributes
        # A seed is not passed here, but reset will be called again by the environment wrapper
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.line_length = self.INITIAL_LINE_LENGTH
        
        # Start the line in the center of the screen
        start_pos = pygame.Vector2(self.WIDTH // 2, self.HEIGHT // 2)
        num_initial_segments = int(self.INITIAL_LINE_LENGTH / self.GROWTH_PER_STEP) + 1
        
        # Use a deque with a maxlen for efficient segment management
        max_segments = int(self.TARGET_LINE_LENGTH / self.GROWTH_PER_STEP) + 2
        self.line_segments = deque([start_pos] * num_initial_segments, maxlen=max_segments)

        # Generate a new set of obstacles for the episode
        self._generate_obstacles()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        movement = action[0]  # 0=none, 1=up, 2=down, 3=left, 4=right
        
        # --- Handle Line Growth ---
        grew = self._handle_growth(movement)
        if grew:
            reward += 0.1  # Continuous feedback for successful growth

        # --- Handle Collisions ---
        collided = self._check_collisions()
        if collided:
            self.line_length = max(0.0, self.line_length - self.SHRINK_ON_COLLISION)
            
            # Remove segments from the tail of the line to reflect new length
            num_segments_to_keep = int(self.line_length / self.GROWTH_PER_STEP) + 1
            while len(self.line_segments) > num_segments_to_keep:
                self.line_segments.popleft()

            reward -= 0.5  # Penalty for collision

        # --- Check Termination Conditions ---
        terminated = False
        truncated = False
        if self.line_length >= self.TARGET_LINE_LENGTH:
            terminated = True
            self.game_over = True
            reward += 100.0  # Large event-based reward for winning
        elif self.line_length <= 0:
            terminated = True
            self.game_over = True
            reward -= 10.0  # Event-based penalty for losing
        elif self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time/step limits
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _handle_growth(self, movement):
        if movement == 0:  # No-op
            return False

        current_head = self.line_segments[-1]
        new_head = pygame.Vector2(current_head)
        
        if movement == 1: new_head.y -= self.GROWTH_PER_STEP  # Up
        elif movement == 2: new_head.y += self.GROWTH_PER_STEP  # Down
        elif movement == 3: new_head.x -= self.GROWTH_PER_STEP  # Left
        elif movement == 4: new_head.x += self.GROWTH_PER_STEP  # Right

        # Boundary check: prevent growing outside the screen
        if not (0 <= new_head.x < self.WIDTH and 0 <= new_head.y < self.HEIGHT):
            return False

        self.line_segments.append(new_head)
        self.line_length += self.GROWTH_PER_STEP
        return True

    def _check_collisions(self):
        if len(self.line_segments) < 2:
            return False

        # Check only the newest point (the head) for collisions
        head = self.line_segments[-1]
        
        for pos, radius in self.obstacles:
            if head.distance_to(pos) < radius:
                return True
        return False

    def _generate_obstacles(self):
        self.obstacles = []
        spawn_margin = self.OBSTACLE_RADIUS + self.OBSTACLE_MIN_DIST
        
        for _ in range(self.NUM_OBSTACLES):
            placed = False
            # Prevent infinite loops if space becomes too crowded
            for _ in range(100): # 100 attempts to place an obstacle
                pos = pygame.Vector2(
                    self.np_random.uniform(spawn_margin, self.WIDTH - spawn_margin),
                    self.np_random.uniform(spawn_margin, self.HEIGHT - spawn_margin)
                )
                
                # Check distance to other obstacles to prevent overlap
                too_close = False
                for other_pos, other_radius in self.obstacles:
                    min_dist_apart = self.OBSTACLE_RADIUS + other_radius + self.OBSTACLE_MIN_DIST
                    if pos.distance_to(other_pos) < min_dist_apart:
                        too_close = True
                        break
                
                if not too_close:
                    self.obstacles.append((pos, self.OBSTACLE_RADIUS))
                    placed = True
                    break
            if not placed:
                # Could not place an obstacle after 100 tries, which is unlikely but possible.
                # In a real game, one might log this, but here we just continue.
                pass

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw obstacles with anti-aliasing for visual quality
        for pos, radius in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), int(radius), self.COLOR_OBSTACLE_OUTLINE)

        # Draw line with a smooth color gradient from tail to head
        num_points = len(self.line_segments)
        if num_points > 1:
            points = list(self.line_segments)
            for i in range(num_points - 1):
                start_pos = points[i]
                end_pos = points[i+1]
                
                # Interpolate color based on position in the line
                ratio = i / max(1, num_points - 2) if num_points > 2 else 1.0
                color = self.COLOR_LINE_TAIL.lerp(self.COLOR_LINE_HEAD, ratio)
                
                pygame.draw.line(self.screen, color, start_pos, end_pos, width=4)

        # Draw a distinct, glowing head for clarity
        if num_points > 0:
            head_pos = self.line_segments[-1]
            pygame.gfxdraw.filled_circle(self.screen, int(head_pos.x), int(head_pos.y), 3, self.COLOR_LINE_HEAD)
            pygame.gfxdraw.aacircle(self.screen, int(head_pos.x), int(head_pos.y), 3, (255, 255, 255))

    def _render_ui(self):
        # Display current line length vs target length
        length_text = self.font.render(f"Length: {int(self.line_length)} / {int(self.TARGET_LINE_LENGTH)}", True, self.COLOR_TEXT)
        self.screen.blit(length_text, (10, 10))

        # Display current steps vs max steps
        steps_text = self.font.render(f"Steps: {self.steps} / {self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 10))

        # Display game over message on a semi-transparent overlay
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            msg = "VICTORY!" if self.line_length >= self.TARGET_LINE_LENGTH else "GAME OVER"
            color = (100, 255, 100) if self.line_length >= self.TARGET_LINE_LENGTH else (255, 100, 100)

            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render(msg, True, color)
            text_rect = game_over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "line_length": self.line_length,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # --- Manual Play Block for Testing and Demonstration ---
    # This block is not used for evaluation but is helpful for development.
    # It requires a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc.
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Line Grower")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0  # Default action: no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset on 'R' key
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the display window
        draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()