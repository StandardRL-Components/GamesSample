
# Generated: 2025-08-28T03:35:29.902769
# Source Brief: brief_02065.md
# Brief Index: 2065

        
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
        "Controls: Use arrow keys to move. Collect all yellow gems and avoid the red mines."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate an isometric maze, collecting gems while avoiding mines to achieve the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 19, 13
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 24, 12
        self.CUBE_HEIGHT = 18
        self.MAX_STAGES = 3
        self.GEMS_PER_STAGE = 15
        self.MAX_STEPS_PER_STAGE = 60

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_WALL_TOP = (60, 68, 87)
        self.COLOR_WALL_LEFT = (48, 54, 70)
        self.COLOR_WALL_RIGHT = (39, 44, 57)
        self.COLOR_WALL_OUTLINE = (30, 34, 43)
        self.COLOR_PLAYER_TOP = (50, 150, 255)
        self.COLOR_PLAYER_LEFT = (40, 120, 204)
        self.COLOR_PLAYER_RIGHT = (30, 90, 153)
        self.COLOR_GEM = (255, 220, 0)
        self.COLOR_GEM_SPARKLE = (255, 255, 180)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_MINE_PULSE = (255, 120, 120)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_SHADOW = (10, 10, 10)
        self.COLOR_STAGE_TEXT = (255, 255, 255)

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
        try:
            self.font_ui = pygame.font.SysFont("dejavusans", 20)
            self.font_stage = pygame.font.SysFont("dejavusans", 28, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_stage = pygame.font.SysFont(None, 36, bold=True)

        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.steps_in_stage = 0
        self.player_pos = [0, 0]
        self.gems = []
        self.mines = []
        self.maze_grid = []
        self.stage_complete = False
        self.rng = None
        
        self._validate_implementation()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.steps_in_stage = 0
        self.stage_complete = False
        
        self.maze_grid = self._generate_maze(self.GRID_WIDTH, self.GRID_HEIGHT)
        
        path_tiles = []
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.maze_grid[r][c] == 0:
                    path_tiles.append((c, r))
        
        self.rng.shuffle(path_tiles)

        player_x, player_y = path_tiles.pop()
        self.player_pos = [player_x, player_y]

        self.gems = []
        for _ in range(min(len(path_tiles), self.GEMS_PER_STAGE)):
            self.gems.append(list(path_tiles.pop()))

        self.mines = []
        num_mines = min(len(path_tiles), self.stage * 2 + 2)
        for _ in range(num_mines):
            self.mines.append(list(path_tiles.pop()))

    def _generate_maze(self, width, height):
        w, h = (width // 2) * 2 + 1, (height // 2) * 2 + 1
        grid = np.ones((h, w), dtype=np.uint8)
        
        start_x, start_y = self.rng.integers(0, w//2)*2, self.rng.integers(0, h//2)*2
        stack = [(start_x, start_y)]
        grid[start_y, start_x] = 0

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 1:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = self.rng.choice(np.array(neighbors), axis=0)
                grid[(y + ny) // 2, (x + nx) // 2] = 0
                grid[ny, nx] = 0
                stack.append((nx, ny))
            else:
                stack.pop()
        
        return grid.tolist()
    
    def step(self, action):
        movement = action[0]
        reward = 0
        self.game_over = False

        old_pos = list(self.player_pos)
        dist_gem_before = self._get_min_dist(self.player_pos, self.gems)
        dist_mine_before = self._get_min_dist(self.player_pos, self.mines)

        if movement != 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy

            if 0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT and self.maze_grid[new_y][new_x] == 0:
                self.player_pos = [new_x, new_y]

        if self.player_pos != old_pos:
            dist_gem_after = self._get_min_dist(self.player_pos, self.gems)
            dist_mine_after = self._get_min_dist(self.player_pos, self.mines)
            
            if dist_gem_after < dist_gem_before: reward += 1.0
            if dist_mine_after < dist_mine_before: reward -= 0.1
        else:
            reward -= 0.05 # Penalty for no-op or hitting a wall

        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10.0
            # sfx: gem_collect

        if self.player_pos in self.mines:
            self.game_over = True
            reward -= 100.0
            # sfx: explosion

        self.steps += 1
        self.steps_in_stage += 1

        if not self.gems:
            self.stage_complete = True
            self.score += 50
            reward += 50.0
            # sfx: stage_clear
            
            if self.stage >= self.MAX_STAGES:
                self.game_over = True # Won the game
            else:
                self.stage += 1
                self._setup_stage()
        
        if self.steps_in_stage >= self.MAX_STEPS_PER_STAGE and not self.game_over and not self.stage_complete:
            self.game_over = True # Timeout
            # sfx: timeout

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_min_dist(self, pos, targets):
        if not targets: return float('inf')
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    def _iso_to_screen(self, x, y):
        screen_x = (x - y) * self.TILE_WIDTH_HALF + self.SCREEN_WIDTH / 2
        screen_y = (x + y) * self.TILE_HEIGHT_HALF + (self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF))
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, x, y, top_color, left_color, right_color):
        sx, sy = self._iso_to_screen(x, y)
        
        # Points for the cube
        p_top = [
            (sx, sy - self.CUBE_HEIGHT),
            (sx + self.TILE_WIDTH_HALF, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF),
            (sx, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF * 2),
            (sx - self.TILE_WIDTH_HALF, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF),
        ]
        p_left = [
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT_HALF * 2),
            (sx, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF * 2),
            (sx - self.TILE_WIDTH_HALF, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF),
        ]
        p_right = [
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + self.TILE_HEIGHT_HALF * 2),
            (sx, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF * 2),
            (sx + self.TILE_WIDTH_HALF, sy - self.CUBE_HEIGHT + self.TILE_HEIGHT_HALF),
        ]
        
        pygame.gfxdraw.filled_polygon(surface, p_top, top_color)
        pygame.gfxdraw.aapolygon(surface, p_top, self.COLOR_WALL_OUTLINE)
        pygame.gfxdraw.filled_polygon(surface, p_left, left_color)
        pygame.gfxdraw.aapolygon(surface, p_left, self.COLOR_WALL_OUTLINE)
        pygame.gfxdraw.filled_polygon(surface, p_right, right_color)
        pygame.gfxdraw.aapolygon(surface, p_right, self.COLOR_WALL_OUTLINE)

    def _render_game(self):
        # Painter's algorithm: draw from back to front
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.maze_grid[r][c] == 1:
                    self._draw_iso_cube(self.screen, c, r, self.COLOR_WALL_TOP, self.COLOR_WALL_LEFT, self.COLOR_WALL_RIGHT)
        
        # Draw dynamic elements
        for mine_pos in self.mines:
            sx, sy = self._iso_to_screen(mine_pos[0], mine_pos[1])
            pulse = (math.sin(self.steps * 0.2) + 1) / 2
            radius = int(self.TILE_WIDTH_HALF * 0.4 + pulse * 3)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_MINE)
            pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, self.COLOR_MINE_PULSE)

        for gem_pos in self.gems:
            sx, sy = self._iso_to_screen(gem_pos[0], gem_pos[1])
            sparkle = (math.sin(self.steps * 0.3 + gem_pos[0]) + 1) / 2
            size = self.TILE_WIDTH_HALF * 0.5 + sparkle * 3
            points = [
                (sx, sy - size * 0.7),
                (sx + size * 0.5, sy),
                (sx, sy + size * 0.7),
                (sx - size * 0.5, sy),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM_SPARKLE)
        
        # Draw player
        self._draw_iso_cube(self.screen, self.player_pos[0], self.player_pos[1], self.COLOR_PLAYER_TOP, self.COLOR_PLAYER_LEFT, self.COLOR_PLAYER_RIGHT)
    
    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow_color=None, shadow_offset=(1, 1)):
            if shadow_color:
                text_surf_shadow = font.render(text, True, shadow_color)
                self.screen.blit(text_surf_shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Score
        draw_text(f"SCORE: {self.score}", self.font_ui, self.COLOR_UI_TEXT, (15, 10), self.COLOR_UI_SHADOW)
        
        # Timer
        time_left = self.MAX_STEPS_PER_STAGE - self.steps_in_stage
        time_text = f"TIME: {time_left}"
        text_w = self.font_ui.size(time_text)[0]
        draw_text(time_text, self.font_ui, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH - text_w - 15, 10), self.COLOR_UI_SHADOW)

        # Stage
        stage_text = f"STAGE {self.stage} / {self.MAX_STAGES}"
        text_w, text_h = self.font_stage.size(stage_text)
        draw_text(stage_text, self.font_stage, self.COLOR_STAGE_TEXT, ((self.SCREEN_WIDTH - text_w) / 2, self.SCREEN_HEIGHT - text_h - 10), self.COLOR_UI_SHADOW)

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
            "stage": self.stage,
            "gems_left": len(self.gems)
        }
    
    def close(self):
        pygame.quit()

    def _validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")