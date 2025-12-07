
# Generated: 2025-08-28T03:33:01.787062
# Source Brief: brief_04960.md
# Brief Index: 4960

        
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
        "Controls: Arrow keys to move your blue square. "
        "Collect all yellow gems to advance to the next stage."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated minefield, collecting gems while avoiding explosive mines. "
        "Complete 3 stages to win. You have a limited number of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and Grid Dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 20, 15
        self.CELL_W = self.WIDTH // self.GRID_W
        self.CELL_H = self.HEIGHT // self.GRID_H
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_GEM = (255, 220, 50)
        self.COLOR_MINE = (255, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        
        # State variables (initialized in reset)
        self.rng = None
        self.player_pos = None
        self.gems = []
        self.mines = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.total_time_steps = 0
        self.current_stage = 0
        self.game_over = False
        self.game_won = False
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        
        self.score = 0
        self.steps = 0
        self.total_time_steps = 1800
        self.current_stage = 1
        self.game_over = False
        self.game_won = False
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.particles = []
        num_mines = 3 + 2 * self.current_stage  # 5, 7, 9
        num_gems = 20

        all_pos = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.rng.shuffle(all_pos)
        
        self.player_pos = list(all_pos.pop())
        self.gems = [list(pos) for pos in all_pos[:num_gems]]
        all_pos = all_pos[num_gems:]
        self.mines = [list(pos) for pos in all_pos[:num_mines]]

    def step(self, action):
        movement = action[0]
        
        self.steps += 1
        self.total_time_steps -= 1
        reward = 0.0
        terminated = False

        old_pos = list(self.player_pos)
        
        if movement == 1: self.player_pos[1] -= 1  # Up
        elif movement == 2: self.player_pos[1] += 1  # Down
        elif movement == 3: self.player_pos[0] -= 1  # Left
        elif movement == 4: self.player_pos[0] += 1  # Right
        
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.GRID_W - 1)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.GRID_H - 1)
        
        new_pos = list(self.player_pos)

        if old_pos != new_pos:
            if self.gems:
                dist_gem_before = self._get_closest_dist(old_pos, self.gems)
                dist_gem_after = self._get_closest_dist(new_pos, self.gems)
                if dist_gem_after < dist_gem_before: reward += 1.0
            
            if self.mines:
                dist_mine_before = self._get_closest_dist(old_pos, self.mines)
                dist_mine_after = self._get_closest_dist(new_pos, self.mines)
                if dist_mine_after < dist_mine_before: reward -= 0.1

        gem_to_remove = next((gem for gem in self.gems if self.player_pos == gem), None)
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            reward += 10
            self.score += 10
            # sfx: gem collect
            self._create_particles(self._grid_to_pixel(gem_to_remove), self.COLOR_GEM, 15)

            if not self.gems:
                reward += 50
                self.score += 50
                self.current_stage += 1
                if self.current_stage > 3:
                    reward += 100
                    self.score += 100
                    terminated = True
                    self.game_won = True
                    # sfx: game win
                else:
                    self._setup_stage()
                    # sfx: stage clear

        if not terminated and any(self.player_pos == mine for mine in self.mines):
            reward = -100
            self.score = max(0, self.score - 100) # Score can't go too negative
            terminated = True
            self.game_over = True
            # sfx: explosion
            self._create_particles(self._grid_to_pixel(self.player_pos), self.COLOR_MINE, 50, 5)
        
        if self.total_time_steps <= 0:
            terminated = True
            self.game_over = True

        self._update_particles()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_closest_dist(self, pos, entity_list):
        if not entity_list: return float('inf')
        return min(abs(pos[0] - e[0]) + abs(pos[1] - e[1]) for e in entity_list)

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_W + self.CELL_W / 2
        py = grid_pos[1] * self.CELL_H + self.CELL_H / 2
        return (px, py)

    def _create_particles(self, pos, color, count, max_speed=3):
        for _ in range(count):
            self.particles.append({
                'pos': list(pos),
                'vel': [self.rng.uniform(-max_speed, max_speed), self.rng.uniform(-max_speed, max_speed)],
                'color': color,
                'life': self.rng.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x in range(self.GRID_W + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_W, 0), (x * self.CELL_W, self.HEIGHT))
        for y in range(self.GRID_H + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_H), (self.WIDTH, y * self.CELL_H))

        for mine in self.mines:
            px, py = self._grid_to_pixel(mine)
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), int(self.CELL_W * 0.35), self.COLOR_MINE)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), int(self.CELL_W * 0.35), self.COLOR_MINE)
        
        for gem in self.gems:
            px, py = self._grid_to_pixel(gem)
            self._draw_diamond(self.screen, self.COLOR_GEM, (px, py), int(self.CELL_W * 0.4))

        if not (self.game_over and not self.game_won):
            px, py = self._grid_to_pixel(self.player_pos)
            player_rect = pygame.Rect(px - self.CELL_W * 0.3, py - self.CELL_H * 0.3, self.CELL_W * 0.6, self.CELL_H * 0.6)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in self.COLOR_PLAYER), player_rect, width=2, border_radius=3)
        
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 40))))
            color = (*p['color'], alpha)
            size = max(1, int(p['life'] / 10))
            temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (size / 2, size / 2, size, size))
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))

    def _draw_diamond(self, surface, color, center, size):
        x, y = center
        points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        int_points = [(int(px), int(py)) for px, py in points]
        pygame.gfxdraw.aapolygon(surface, int_points, color)
        pygame.gfxdraw.filled_polygon(surface, int_points, color)

    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        stage_text = self.font_medium.render(f"Stage: {self.current_stage}/3", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH - stage_text.get_width() - 10, 10))
        
        time_left = max(0, self.total_time_steps)
        timer_text = self.font_medium.render(f"Moves: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 10))

        if self.game_over or self.game_won:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            text_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, bg_rect.topleft)
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "gems_left": len(self.gems),
            "moves_left": self.total_time_steps,
        }
    
    def validate_implementation(self):
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