
# Generated: 2025-08-27T18:07:05.379207
# Source Brief: brief_01734.md
# Brief Index: 1734

        
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
        "Controls: Use arrow keys (up, down, left, right) to move your character on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric grid to collect all the gems before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 12
        
        # Game Parameters
        self.MAX_STEPS = 60
        self.NUM_GEMS = 15
        
        # Isometric Tile Dimensions
        self.TILE_WIDTH = 40
        self.TILE_HEIGHT = 20
        self.TILE_WIDTH_HALF = self.TILE_WIDTH // 2
        self.TILE_HEIGHT_HALF = self.TILE_HEIGHT // 2
        
        # Grid Origin (to center it on screen)
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 80
        
        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_FILL = (35, 38, 46)
        self.COLOR_GRID_OUTLINE = (55, 60, 72)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255, 40) # RGBA
        self.GEM_COLORS = [
            (255, 87, 87),   # Red
            (87, 255, 150),  # Green
            (87, 150, 255),  # Blue
            (255, 255, 87),  # Yellow
            (200, 87, 255),  # Purple
        ]
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_TIMER_BAR_BG = (50, 50, 50)
        self.COLOR_TIMER_GREEN = (0, 200, 0)
        self.COLOR_TIMER_YELLOW = (200, 200, 0)
        self.COLOR_TIMER_RED = (200, 0, 0)
        
        # Gym Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24)
            self.font_ui_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 30)
            self.font_ui_small = pygame.font.SysFont(None, 24)

        # Game State (initialized in reset)
        self.player_pos = (0, 0)
        self.gems = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        self.validate_implementation()
    
    def _grid_to_iso(self, gx, gy):
        """Converts grid coordinates to isometric screen coordinates."""
        screen_x = self.ORIGIN_X + (gx - gy) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (gx + gy) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _get_dist_to_nearest_gem(self):
        if not self.gems:
            return 0
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for gem in self.gems:
            gem_x, gem_y = gem["pos"]
            dist = abs(player_x - gem_x) + abs(player_y - gem_y) # Manhattan distance
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        
        self.gems = []
        possible_locations = []
        for gx in range(self.GRID_WIDTH):
            for gy in range(self.GRID_HEIGHT):
                if (gx, gy) != self.player_pos:
                    possible_locations.append((gx, gy))
        
        gem_indices = self.np_random.choice(len(possible_locations), self.NUM_GEMS, replace=False)
        for i in gem_indices:
            pos = possible_locations[i]
            color = random.choice(self.GEM_COLORS)
            self.gems.append({"pos": pos, "color": color})
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        dist_before = self._get_dist_to_nearest_gem()
        
        px, py = self.player_pos
        if movement == 1: py -= 1  # Up (isometric up-left)
        elif movement == 2: py += 1  # Down (isometric down-right)
        elif movement == 3: px -= 1  # Left (isometric down-left)
        elif movement == 4: px += 1  # Right (isometric up-right)
        
        px = max(0, min(self.GRID_WIDTH - 1, px))
        py = max(0, min(self.GRID_HEIGHT - 1, py))
        self.player_pos = (px, py)
        
        dist_after = self._get_dist_to_nearest_gem()

        if dist_after < dist_before: reward += 1
        elif dist_after > dist_before: reward -= 1
            
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem["pos"]:
                gem_to_remove = gem
                break
        
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            self.score += 1
            reward += 10 # sfx: gem_collect.wav
            
        self.steps += 1
        
        won = len(self.gems) == 0
        lost = self.steps >= self.MAX_STEPS
        terminated = won or lost
        
        if won:
            reward += 50 # sfx: victory.wav
            self.game_over = True
        elif lost:
            reward -= 50 # sfx: game_over.wav
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _render_grid(self):
        for gy in range(self.GRID_HEIGHT):
            for gx in range(self.GRID_WIDTH):
                sx, sy = self._grid_to_iso(gx, gy)
                points = [
                    (sx, sy - self.TILE_HEIGHT_HALF),
                    (sx + self.TILE_WIDTH_HALF, sy),
                    (sx, sy + self.TILE_HEIGHT_HALF),
                    (sx - self.TILE_WIDTH_HALF, sy)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GRID_FILL)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID_OUTLINE)

    def _render_gems(self):
        for gem in sorted(self.gems, key=lambda g: g['pos'][0] + g['pos'][1]):
            sx, sy = self._grid_to_iso(gem["pos"][0], gem["pos"][1])
            color = gem["color"]
            highlight_color = tuple(min(255, c + 80) for c in color)
            
            radius = self.TILE_HEIGHT_HALF - 3
            pygame.gfxdraw.aacircle(self.screen, sx, int(sy - 2), radius, color)
            pygame.gfxdraw.filled_circle(self.screen, sx, int(sy - 2), radius, color)
            
            highlight_radius = radius // 3
            pygame.gfxdraw.aacircle(self.screen, sx - 3, int(sy - 5), highlight_radius, highlight_color)
            pygame.gfxdraw.filled_circle(self.screen, sx - 3, int(sy - 5), highlight_radius, highlight_color)

    def _render_player(self):
        sx, sy = self._grid_to_iso(self.player_pos[0], self.player_pos[1])
        radius = self.TILE_HEIGHT_HALF
        
        glow_radius = radius + 8
        glow_surf = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (sx - glow_radius, sy - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.aacircle(self.screen, sx, int(sy - 4), radius, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, sx, int(sy - 4), radius, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        
        progress = max(0, 1.0 - (self.steps / self.MAX_STEPS))
        fill_width = int(bar_width * progress)
        
        if progress > 0.6: color = self.COLOR_TIMER_GREEN
        elif progress > 0.3: color = self.COLOR_TIMER_YELLOW
        else: color = self.COLOR_TIMER_RED
            
        if fill_width > 0:
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
        
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        moves_left = self.MAX_STEPS - self.steps
        moves_text = self.font_ui_small.render(f"Moves Left: {moves_left}", True, self.COLOR_UI_TEXT)
        text_rect = moves_text.get_rect(center=(bar_x + bar_width / 2, bar_y + bar_height / 2))
        self.screen.blit(moves_text, text_rect)
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # Render entities in order for correct layering
        entities = []
        for gem in self.gems:
            entities.append({'pos': gem['pos'], 'type': 'gem', 'data': gem})
        entities.append({'pos': self.player_pos, 'type': 'player', 'data': None})

        # Sort by isometric depth (y+x)
        entities.sort(key=lambda e: e['pos'][0] + e['pos'][1])

        self._render_grid()
        for entity in entities:
            if entity['type'] == 'gem':
                self._render_single_gem(entity['data'])
            elif entity['type'] == 'player':
                self._render_player()

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_single_gem(self, gem):
        sx, sy = self._grid_to_iso(gem["pos"][0], gem["pos"][1])
        color = gem["color"]
        highlight_color = tuple(min(255, c + 80) for c in color)
        
        radius = self.TILE_HEIGHT_HALF - 3
        pygame.gfxdraw.aacircle(self.screen, sx, int(sy - 2), radius, color)
        pygame.gfxdraw.filled_circle(self.screen, sx, int(sy - 2), radius, color)
        
        highlight_radius = radius // 3
        pygame.gfxdraw.aacircle(self.screen, sx - 3, int(sy - 5), highlight_radius, highlight_color)
        pygame.gfxdraw.filled_circle(self.screen, sx - 3, int(sy - 5), highlight_radius, highlight_color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_left": len(self.gems),
            "player_pos": self.player_pos,
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")