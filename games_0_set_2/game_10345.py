import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:13:41.881893
# Source Brief: brief_00345.md
# Brief Index: 345
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Stack falling blocks to build the tallest tower possible. "
        "Move the platform left and right to catch the blocks."
    )
    user_guide = "Controls: Use the ← and → arrow keys to move the platform."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.WALL_WIDTH = 10
        self.PLAY_AREA_WIDTH = self.SCREEN_WIDTH - 2 * self.WALL_WIDTH
        
        # Gameplay Constants
        self.MAX_STEPS = 2000
        self.WIN_SCORE = 100
        self.PLATFORM_HEIGHT = 15
        self.PLATFORM_WIDTH = 120
        self.PLATFORM_SPEED = 8
        self.INITIAL_FALL_SPEED = 2.0
        self.FALL_SPEED_INCREMENT = 0.05
        
        # Colors (Vibrant, high-contrast)
        self.COLOR_BG_TOP = (15, 20, 40)
        self.COLOR_BG_BOTTOM = (5, 5, 15)
        self.COLOR_WALL = (60, 60, 80, 100) # Semi-transparent
        self.COLOR_PLATFORM = (230, 230, 255)
        self.COLOR_PLATFORM_OUTLINE = (150, 150, 180)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)
        self.BLOCK_BASE_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (255, 128, 0),  # Orange
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.platform = None
        self.falling_block = None
        self.stacked_blocks = []
        self.steps = 0
        self.score = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.game_over = False
        
        # Initialize state
        # self.reset() # reset is called by the agent/runner

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        
        # Reset platform
        platform_y = self.SCREEN_HEIGHT - self.PLATFORM_HEIGHT
        platform_x = (self.SCREEN_WIDTH - self.PLATFORM_WIDTH) / 2
        self.platform = pygame.Rect(platform_x, platform_y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        
        # Reset blocks
        self.stacked_blocks = []
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are not used in this game
        
        reward = 0.0
        
        if not self.game_over:
            # --- 1. Update Player Action ---
            if movement == 3:  # Left
                self.platform.x -= self.PLATFORM_SPEED
            elif movement == 4:  # Right
                self.platform.x += self.PLATFORM_SPEED
            
            # Clamp platform position within play area
            self.platform.x = max(self.WALL_WIDTH, min(self.platform.x, self.SCREEN_WIDTH - self.WALL_WIDTH - self.PLATFORM_WIDTH))
            
            # --- 2. Update Game Logic ---
            # Move falling block
            self.falling_block['rect'].y += self.fall_speed
            
            # Check for landing
            landing_y = self._get_landing_y(self.falling_block['rect'])
            if self.falling_block['rect'].bottom >= landing_y:
                # Snap block to landing position
                self.falling_block['rect'].bottom = landing_y
                
                # Check if landed block is out of bounds
                if self.falling_block['rect'].left < self.WALL_WIDTH or self.falling_block['rect'].right > self.SCREEN_WIDTH - self.WALL_WIDTH:
                    # Block landed but slid off the side
                    reward -= 5.0
                    self.game_over = True
                else:
                    # Successful catch
                    self.stacked_blocks.append(self.falling_block)
                    self.score += 1
                    reward += 0.1
                    
                    # Difficulty scaling
                    if self.score > 0 and self.score % 50 == 0:
                        self.fall_speed += self.FALL_SPEED_INCREMENT

                    # Check for win condition
                    if self.score >= self.WIN_SCORE:
                        reward += 100.0
                        self.game_over = True
                    else:
                        self._spawn_new_block()

            # Check for loss condition (dropped block)
            if self.falling_block['rect'].top > self.SCREEN_HEIGHT:
                reward -= 5.0
                self.game_over = True

        # --- 3. Finalize Step ---
        self.steps += 1
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _spawn_new_block(self):
        min_w, max_w = 30, 100
        min_h, max_h = 20, 50
        width = self.np_random.integers(min_w, max_w + 1)
        height = self.np_random.integers(min_h, max_h + 1)
        
        x_pos = self.np_random.integers(
            self.WALL_WIDTH, 
            self.SCREEN_WIDTH - self.WALL_WIDTH - width + 1
        )
        
        # Color based on size (area)
        max_area = max_w * max_h
        area = width * height
        brightness_factor = 0.6 + 0.4 * (1 - area / max_area) # Larger blocks are darker
        
        base_color = self.BLOCK_BASE_COLORS[self.np_random.integers(len(self.BLOCK_BASE_COLORS))]
        color = tuple(min(255, int(c * brightness_factor)) for c in base_color)
        
        self.falling_block = {
            'rect': pygame.Rect(x_pos, -height, width, height),
            'color': color,
            'outline_color': tuple(max(0, c-50) for c in color)
        }

    def _get_landing_y(self, block_rect):
        """Calculates the Y-coordinate where a block would land."""
        possible_surfaces = [self.platform] + [b['rect'] for b in self.stacked_blocks]
        landing_y = self.platform.top
        
        highest_surface_y = self.SCREEN_HEIGHT + 100 # A large number

        for surface in possible_surfaces:
            # Check for horizontal overlap
            if block_rect.left < surface.right and block_rect.right > surface.left:
                # Is this surface above the current highest?
                if surface.top < highest_surface_y:
                    highest_surface_y = surface.top

        return highest_surface_y

    def _get_observation(self):
        # Render all elements to the surface
        self._render_background()
        self._render_game()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        """Draws a vertical gradient for the background."""
        self.screen.fill(self.COLOR_BG_BOTTOM)
        top_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_BG_TOP, top_rect, 0, border_bottom_left_radius=200, border_bottom_right_radius=200)


    def _render_game(self):
        """Renders all game objects like walls, platform, and blocks."""
        # Walls
        wall_surface = pygame.Surface((self.WALL_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        wall_surface.fill(self.COLOR_WALL)
        self.screen.blit(wall_surface, (0, 0))
        self.screen.blit(wall_surface, (self.SCREEN_WIDTH - self.WALL_WIDTH, 0))
        
        # Stacked blocks
        for block in self.stacked_blocks:
            pygame.gfxdraw.box(self.screen, block['rect'], block['color'])
            pygame.gfxdraw.rectangle(self.screen, block['rect'], block['outline_color'])
            
        # Falling block
        if self.falling_block:
            fb = self.falling_block
            pygame.gfxdraw.box(self.screen, fb['rect'], fb['color'])
            pygame.gfxdraw.rectangle(self.screen, fb['rect'], fb['outline_color'])

        # Platform
        pygame.gfxdraw.box(self.screen, self.platform, self.COLOR_PLATFORM)
        pygame.gfxdraw.rectangle(self.screen, self.platform, self.COLOR_PLATFORM_OUTLINE)

    def _render_text(self, text, font, color, position, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (position[0] + 2, position[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, position)

    def _render_ui(self):
        """Renders the score and other UI elements."""
        # Score (Tower Height)
        score_text = f"Height: {self.score}"
        self._render_text(score_text, self.font_medium, self.COLOR_TEXT, (20, 10))

        # Game Over / Win Message
        if self.game_over:
            if self.score >= self.WIN_SCORE:
                msg = "TOWER COMPLETE!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_surf = self.font_large.render(msg, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(shadow_surf, (text_rect.x + 3, text_rect.y + 3))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fall_speed": self.fall_speed,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be run by the autograder
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Builder")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    running = True
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False

        if not terminated and not truncated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS for smooth human playback
        
    env.close()