
# Generated: 2025-08-27T17:12:26.469510
# Source Brief: brief_01456.md
# Brief Index: 1456

        
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
    user_guide = "Controls: ↑↓←→ to move your character one square at a time."

    # Must be a short, user-facing description of the game:
    game_description = "Collect 10 gems on the grid before the 120-second timer runs out. Each move costs one second."

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255, 50)
    COLOR_TEXT = (220, 220, 240)
    GEM_COLORS = [(255, 80, 80), (255, 255, 80), (80, 255, 80), (80, 150, 255)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Grid layout calculation
        self.cell_size = min(self.SCREEN_WIDTH // (self.GRID_SIZE + 2), self.SCREEN_HEIGHT // (self.GRID_SIZE + 2))
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2
        
        # Initialize state variables
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.player_pos = None
        self.gems = []
        self.time_remaining = 0
        self.gems_collected = 0
        self.gems_to_win = 10
        self.max_time = 120
        self.particles = []
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_remaining = self.max_time
        self.gems_collected = 0
        self.particles = []
        
        # Place player in the center
        self.player_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        
        # Generate gem positions
        self.gems = []
        possible_coords = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        
        player_start_tuple = tuple(self.player_pos)
        if player_start_tuple in possible_coords:
            possible_coords.remove(player_start_tuple)
        
        num_gems = min(self.gems_to_win, len(possible_coords))
        gem_indices = self.np_random.choice(len(possible_coords), num_gems, replace=False)
        
        for i in gem_indices:
            self.gems.append(np.array(possible_coords[i]))
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        reward = 0.0
        
        dist_before = self._find_nearest_gem_dist()
        
        prev_pos = self.player_pos.copy()
        if movement == 1: self.player_pos[1] -= 1
        elif movement == 2: self.player_pos[1] += 1
        elif movement == 3: self.player_pos[0] -= 1
        elif movement == 4: self.player_pos[0] += 1
        
        self.player_pos = np.clip(self.player_pos, 0, self.GRID_SIZE - 1)
        
        if movement != 0:
            self.time_remaining -= 1
        
        dist_after = self._find_nearest_gem_dist()
        
        if dist_after < dist_before: reward += 1.0
        elif dist_after > dist_before: reward -= 0.1
        
        gem_to_remove = -1
        for i, gem_pos in enumerate(self.gems):
            if np.array_equal(self.player_pos, gem_pos):
                self._create_particles(gem_pos, self.GEM_COLORS[i % len(self.GEM_COLORS)])
                gem_to_remove = i
                self.gems_collected += 1
                reward += 10.0
                # SFX: Gem collect sound
                break
        
        if gem_to_remove != -1:
            self.gems.pop(gem_to_remove)

        terminated = False
        if self.gems_collected >= self.gems_to_win:
            reward += 50.0
            terminated = True
            self.game_over = True
            # SFX: Win fanfare
        elif self.time_remaining <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
            # SFX: Loss buzzer
            
        self.score += reward
        self.steps += 1
        
        self._update_particles()
        
        return (self._get_observation(), reward, terminated, False, self._get_info())
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid()
        self._render_gems()
        self._render_particles()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gems_collected": self.gems_collected,
            "time_remaining": self.time_remaining
        }

    def _grid_to_pixel(self, grid_pos):
        px = self.grid_offset_x + grid_pos[0] * self.cell_size + self.cell_size // 2
        py = self.grid_offset_y + grid_pos[1] * self.cell_size + self.cell_size // 2
        return int(px), int(py)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            start_x_v, end_x_v = self.grid_offset_x + i * self.cell_size, self.grid_offset_x + i * self.cell_size
            start_y_v, end_y_v = self.grid_offset_y, self.grid_offset_y + self.grid_height
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x_v, start_y_v), (end_x_v, end_y_v), 1)

            start_x_h, end_x_h = self.grid_offset_x, self.grid_offset_x + self.grid_width
            start_y_h, end_y_h = self.grid_offset_y + i * self.cell_size, self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (start_x_h, start_y_h), (end_x_h, end_y_h), 1)

    def _render_player(self):
        px, py = self._grid_to_pixel(self.player_pos)
        size = int(self.cell_size * 0.7)
        glow_size = int(size * 1.5)
        
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=4)
        self.screen.blit(glow_surf, (px - glow_size // 2, py - glow_size // 2))

        player_rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
    def _render_gems(self):
        base_radius = int(self.cell_size * 0.35)
        for i, gem_pos in enumerate(self.gems):
            px, py = self._grid_to_pixel(gem_pos)
            pulse = math.sin(self.steps * 0.2 + i)
            sparkle_radius = base_radius + int(pulse * 3)
            color = self.GEM_COLORS[i % len(self.GEM_COLORS)]
            
            pygame.gfxdraw.filled_circle(self.screen, px, py, sparkle_radius, color)
            pygame.gfxdraw.aacircle(self.screen, px, py, sparkle_radius, color)

            sparkle_color = tuple(min(255, c + 50) for c in color)
            sparkle_pos_angle = self.steps * 0.1 + i * 2
            sparkle_offset_x = math.cos(sparkle_pos_angle) * sparkle_radius * 0.5
            sparkle_offset_y = math.sin(sparkle_pos_angle) * sparkle_radius * 0.5
            pygame.gfxdraw.filled_circle(self.screen, int(px + sparkle_offset_x), int(py + sparkle_offset_y), 2, sparkle_color)

    def _render_ui(self):
        time_surf = self.font_ui.render(f"TIME: {self.time_remaining}", True, self.COLOR_TEXT)
        gems_surf = self.font_ui.render(f"GEMS: {self.gems_collected} / {self.gems_to_win}", True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (20, 15))
        self.screen.blit(gems_surf, (20, 40))
        
        if self.game_over:
            message, color = ("YOU WIN!", (100, 255, 100)) if self.gems_collected >= self.gems_to_win else ("TIME'S UP!", (255, 100, 100))
            end_surf = self.font_game_over.render(message, True, color)
            end_rect = end_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            bg_rect = end_rect.inflate(40, 20)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 150))
            self.screen.blit(bg_surf, bg_rect.topleft)
            self.screen.blit(end_surf, end_rect)

    def _create_particles(self, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(20):
            self.particles.append({
                'pos': np.array([float(px), float(py)]),
                'vel': np.array([self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]),
                'lifespan': self.np_random.uniform(15, 30),
                'max_lifespan': 30,
                'radius': self.np_random.uniform(2, 5),
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _render_particles(self):
        for p in self.particles:
            life_ratio = max(0, p['lifespan'] / p['max_lifespan'])
            current_radius = int(p['radius'] * life_ratio)
            if current_radius <= 0: continue
            
            current_color = tuple(int(c * life_ratio) for c in p['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), current_radius, current_color)

    def _find_nearest_gem_dist(self):
        if not self.gems: return float('inf')
        distances = [self._manhattan_distance(self.player_pos, gem_pos) for gem_pos in self.gems]
        return min(distances) if distances else float('inf')

    @staticmethod
    def _manhattan_distance(pos1, pos2):
        return np.sum(np.abs(pos1 - pos2))

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")