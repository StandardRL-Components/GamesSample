
# Generated: 2025-08-27T20:34:21.912818
# Source Brief: brief_02506.md
# Brief Index: 2506

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to jump between tiles. Reach the goal before the tiles disappear or you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate a grid of crumbling tiles to reach the goal. Each jump weakens the tile you land on. Plan your path carefully!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.GRID_SIZE = 5
        self.MAX_STEPS = 15
        self.WIDTH, self.HEIGHT = 640, 400

        # --- Colors (Clean, High-Contrast Palette) ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID_BG = (40, 40, 60)
        self.COLOR_TILE_STABLE = (60, 180, 70)  # Green
        self.COLOR_TILE_WARN = (255, 225, 25)  # Yellow
        self.COLOR_TILE_DANGER = (230, 25, 75)  # Red
        self.COLOR_TILE_GONE = (80, 80, 100)  # Grey
        self.COLOR_PLAYER = (50, 200, 255)  # Bright Cyan
        self.COLOR_PLAYER_GLOW = (50, 200, 255, 50)
        self.COLOR_GOAL = (255, 165, 0)  # Orange
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Grid Rendering Calculation
        grid_area_size = self.HEIGHT - 40
        self.tile_size = grid_area_size // self.GRID_SIZE
        self.tile_spacing = self.tile_size // 6
        grid_total_size = self.GRID_SIZE * self.tile_size + (self.GRID_SIZE - 1) * self.tile_spacing
        self.grid_offset_x = (self.WIDTH - grid_total_size) // 2
        self.grid_offset_y = (self.HEIGHT - grid_total_size) // 2

        # Initialize state variables
        self.player_pos = None
        self.goal_pos = None
        self.tile_health = None
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.np_random = None
        
        self.validate_implementation()
    
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # Reset first to initialize necessary variables for observation
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.tile_health = np.full((self.GRID_SIZE, self.GRID_SIZE), 3, dtype=int)
        
        start_x = self.np_random.integers(0, self.GRID_SIZE)
        start_y = self.np_random.integers(0, self.GRID_SIZE)
        self.player_pos = np.array([start_x, start_y])

        while True:
            goal_x = self.np_random.integers(0, self.GRID_SIZE)
            goal_y = self.np_random.integers(0, self.GRID_SIZE)
            self.goal_pos = np.array([goal_x, goal_y])
            if not np.array_equal(self.player_pos, self.goal_pos):
                break
        
        # Degrade the starting tile as the player is already on it
        self.tile_health[self.player_pos[1], self.player_pos[0]] -= 1

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        movement = action[0]
        terminated = False
        reward = -0.1  # Cost per step

        self.steps += 1
        
        if movement != 0:  # 0 is no-op
            move_vec = {1: [0, -1], 2: [0, 1], 3: [-1, 0], 4: [1, 0]}.get(movement, [0, 0])
            next_pos = self.player_pos + move_vec
            
            px, py = next_pos[0], next_pos[1]
            if not (0 <= px < self.GRID_SIZE and 0 <= py < self.GRID_SIZE):
                # Fell off the grid
                reward -= 50
                terminated = True
                self.game_over = True
            elif self.tile_health[py, px] <= 0:
                # Jumped to a disappeared tile
                reward -= 50
                terminated = True
                self.game_over = True
            else:
                # Valid move
                self.player_pos = next_pos
                
                # Reward based on tile health *before* degrading
                current_tile_health = self.tile_health[py, px]
                if current_tile_health == 2: reward += 5  # Was Yellow
                elif current_tile_health == 1: reward += 10 # Was Red
                
                # Degrade the tile
                self.tile_health[py, px] -= 1
                
                if np.array_equal(self.player_pos, self.goal_pos):
                    reward += 100
                    terminated = True
                    self.game_over = True
        
        if not terminated and self.steps >= self.MAX_STEPS:
            reward -= 25
            terminated = True
            self.game_over = True

        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _render_game(self):
        # Draw grid background
        bg_rect = pygame.Rect(self.grid_offset_x - 10, self.grid_offset_y - 10,
                              self.GRID_SIZE * (self.tile_size + self.tile_spacing) - self.tile_spacing + 20,
                              self.GRID_SIZE * (self.tile_size + self.tile_spacing) - self.tile_spacing + 20)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, bg_rect, border_radius=15)
        
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                health = self.tile_health[y, x]
                color = self.COLOR_TILE_GONE
                if health == 3: color = self.COLOR_TILE_STABLE
                elif health == 2: color = self.COLOR_TILE_WARN
                elif health == 1: color = self.COLOR_TILE_DANGER
                
                tile_x = self.grid_offset_x + x * (self.tile_size + self.tile_spacing)
                tile_y = self.grid_offset_y + y * (self.tile_size + self.tile_spacing)
                
                tile_rect = pygame.Rect(tile_x, tile_y, self.tile_size, self.tile_size)
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=5)
                
                if np.array_equal(self.goal_pos, [x, y]):
                    self._draw_star(self.screen, tile_rect.center, self.COLOR_GOAL, 5, self.tile_size * 0.4, self.tile_size * 0.2)

        player_cx = self.grid_offset_x + self.player_pos[0] * (self.tile_size + self.tile_spacing) + self.tile_size // 2
        player_cy = self.grid_offset_y + self.player_pos[1] * (self.tile_size + self.tile_spacing) + self.tile_size // 2
        
        glow_radius = int(self.tile_size * 0.5)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (player_cx - glow_radius, player_cy - glow_radius))
        
        player_radius = int(self.tile_size * 0.3)
        pygame.gfxdraw.filled_circle(self.screen, int(player_cx), int(player_cy), player_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, int(player_cx), int(player_cy), player_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = f"Score: {self.score:.1f}"
        self._draw_text(score_text, (20, 15), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        steps_left = max(0, self.MAX_STEPS - self.steps)
        steps_text = f"Moves Left: {steps_left}"
        text_width = self.font_small.size(steps_text)[0]
        self._draw_text(steps_text, (self.WIDTH - text_width - 20, 15), self.font_small, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        
        if self.game_over:
            message = ""
            if np.array_equal(self.player_pos, self.goal_pos):
                message = "GOAL REACHED!"
            elif self.steps >= self.MAX_STEPS:
                message = "OUT OF MOVES"
            else:
                message = "YOU FELL!"
            
            self._draw_text(message, (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, self.COLOR_GOAL, self.COLOR_TEXT_SHADOW, center=True)

    def _draw_text(self, text, pos, font, color, shadow_color, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def _draw_star(self, surface, center, color, points, outer_radius, inner_radius):
        star_points = []
        angle_step = 360 / (points * 2)
        for i in range(points * 2):
            angle = math.radians(i * angle_step - 90)
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            star_points.append((int(x), int(y)))
        pygame.gfxdraw.filled_polygon(surface, star_points, color)
        pygame.gfxdraw.aapolygon(surface, star_points, color)

    def close(self):
        pygame.font.quit()
        pygame.quit()