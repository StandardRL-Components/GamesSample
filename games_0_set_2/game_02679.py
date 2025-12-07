
# Generated: 2025-08-28T05:37:54.531562
# Source Brief: brief_02679.md
# Brief Index: 2679

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys (↑↓←→) to move your character."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated minefield to collect all gems within a time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.NUM_GEMS = 15
        self.NUM_MINES = 3
        self.MAX_STEPS = 200

        # Visuals
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (255, 220, 50)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_GEM = (50, 200, 255)
        self.COLOR_GEM_GLOW = (50, 200, 255, 50)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_MINE_OUTLINE = (255, 120, 120)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SCORE_TEXT = (255, 255, 100)
        
        # Calculate rendering geometry
        self.CELL_SIZE = 36
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # --- Game State (initialized in reset) ---
        self.player_pos = None
        self.gems = None
        self.mines = None
        self.steps_remaining = None
        self.score = None
        self.last_action = None
        self.game_over = None
        self.rng = None
        
        # Initialize state variables
        self.reset()

    def _generate_level(self):
        """Generates a new level, ensuring all gems are reachable."""
        while True:
            all_cells = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
            self.rng.shuffle(all_cells)
            
            player_pos = all_cells.pop()
            
            mines = []
            for _ in range(self.NUM_MINES):
                mines.append(all_cells.pop())
            
            gems = []
            for _ in range(self.NUM_GEMS):
                gems.append(all_cells.pop())
                
            # --- Pathfinding validation (BFS) to ensure solvability ---
            q = deque([player_pos])
            visited = {player_pos}
            mine_set = set(mines)
            
            while q:
                cx, cy = q.popleft()
                
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = cx + dx, cy + dy
                    
                    if 0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and (nx, ny) not in visited and (nx, ny) not in mine_set:
                        visited.add((nx, ny))
                        q.append((nx, ny))
            
            # Check if all gems are reachable
            all_gems_reachable = all(gem in visited for gem in gems)
            
            if all_gems_reachable:
                self.player_pos = list(player_pos)
                self.gems = [list(g) for g in gems]
                self.mines = [list(m) for m in mines]
                return # Found a valid level

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.rng is None:
            self.rng = np.random.default_rng()

        self._generate_level()
        
        self.steps_remaining = self.MAX_STEPS
        self.score = 0
        self.game_over = False
        self.last_action = 0 # No-op
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        # space_held and shift_held are ignored as per the brief
        
        reward = 0
        terminated = False
        
        # --- Continuous Rewards ---
        dist_before = self._get_distance_to_nearest_gem()
        
        if movement != 0 and movement == self.last_action:
            reward -= 0.1 # Penalize repeating non-noop actions
        self.last_action = movement

        # --- Update Player Position ---
        if movement != 0: # 0 is no-op
            px, py = self.player_pos
            if movement == 1: py -= 1 # Up
            elif movement == 2: py += 1 # Down
            elif movement == 3: px -= 1 # Left
            elif movement == 4: px += 1 # Right
            
            # Clamp to grid boundaries
            self.player_pos[0] = max(0, min(self.GRID_SIZE - 1, px))
            self.player_pos[1] = max(0, min(self.GRID_SIZE - 1, py))

        dist_after = self._get_distance_to_nearest_gem()

        if dist_after is not None and dist_before is not None:
            if dist_after < dist_before:
                reward += 0.1 # Moved closer
            else:
                reward -= 0.02 # Moved further or same distance

        # --- Event-based Rewards & State Changes ---
        if self.player_pos in self.gems:
            self.gems.remove(self.player_pos)
            self.score += 10
            reward += 10
            # Sound placeholder: gem_collect.wav

        if self.player_pos in self.mines:
            reward -= 100
            terminated = True
            # Sound placeholder: explosion.wav

        # --- Termination Conditions ---
        self.steps_remaining -= 1
        if self.steps_remaining <= 0 and not terminated:
            terminated = True

        if not self.gems and not terminated: # All gems collected (Win condition)
            time_bonus = 50 * (self.steps_remaining / self.MAX_STEPS)
            self.score += int(time_bonus)
            reward += 50 + time_bonus
            terminated = True
            # Sound placeholder: win_level.wav
            
        if terminated:
            self.game_over = True

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_distance_to_nearest_gem(self):
        if not self.gems:
            return None
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        for gem_x, gem_y in self.gems:
            dist = abs(player_x - gem_x) + abs(player_y - gem_y) # Manhattan distance
            min_dist = min(min_dist, dist)
        return min_dist

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
        self._render_grid()
        self._render_mines()
        self._render_gems()
        self._render_player()

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y),
                             (self.OFFSET_X + i * self.CELL_SIZE, self.OFFSET_Y + self.GRID_HEIGHT))
            # Horizontal lines
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.OFFSET_X, self.OFFSET_Y + i * self.CELL_SIZE),
                             (self.OFFSET_X + self.GRID_WIDTH, self.OFFSET_Y + i * self.CELL_SIZE))

    def _cell_to_pixel(self, x, y):
        return (self.OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2)

    def _render_mines(self):
        for mx, my in self.mines:
            cx, cy = self._cell_to_pixel(mx, my)
            size = self.CELL_SIZE * 0.35
            points = [
                (cx, cy - size),
                (cx - size, cy + size * 0.7),
                (cx + size, cy + size * 0.7)
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_MINE_OUTLINE)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_MINE)

    def _render_gems(self):
        radius = self.CELL_SIZE * 0.3
        glow_radius = radius * 1.8
        
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_GEM_GLOW, (glow_radius, glow_radius), glow_radius)

        for gx, gy in self.gems:
            cx, cy = self._cell_to_pixel(gx, gy)
            self.screen.blit(glow_surface, (cx - glow_radius, cy - glow_radius))
            pygame.gfxdraw.aacircle(self.screen, int(cx), int(cy), int(radius), self.COLOR_GEM)
            pygame.gfxdraw.filled_circle(self.screen, int(cx), int(cy), int(radius), self.COLOR_GEM)

    def _render_player(self):
        px, py = self._cell_to_pixel(self.player_pos[0], self.player_pos[1])
        size = self.CELL_SIZE * 0.8
        half_size = size / 2
        player_rect = pygame.Rect(px - half_size, py - half_size, size, size)
        
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(4, 4), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _render_ui(self):
        gem_text = f"Gems: {self.NUM_GEMS - len(self.gems)}/{self.NUM_GEMS}"
        gem_surf = self.font_medium.render(gem_text, True, self.COLOR_TEXT)
        self.screen.blit(gem_surf, (20, 15))

        time_text = f"Steps: {self.steps_remaining}"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_surf, time_rect)

        score_text = f"Score: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_SCORE_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(score_surf, score_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if not self.gems: end_text = "LEVEL COMPLETE!"
            elif self.player_pos in self.mines: end_text = "HIT A MINE!"
            else: end_text = "TIME UP!"
            
            end_surf = self.font_large.render(end_text, True, (255, 255, 255))
            end_rect = end_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.MAX_STEPS - self.steps_remaining,
            "gems_collected": self.NUM_GEMS - len(self.gems)
        }
        
    def close(self):
        pygame.quit()