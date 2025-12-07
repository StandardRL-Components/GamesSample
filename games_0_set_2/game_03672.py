
# Generated: 2025-08-28T00:05:25.830735
# Source Brief: brief_03672.md
# Brief Index: 3672

        
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

    user_guide = (
        "Controls: Use arrow keys to move the selector. "
        "Press SPACE to shift the selected column UP. "
        "Press SHIFT to shift the selected row RIGHT."
    )

    game_description = (
        "A turn-based puzzle game. Navigate an isometric cavern, "
        "strategically shifting rows and columns of crystals to "
        "create matches of 3 or more. Clear 20 crystals in 5 moves to win."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 8, 8
    WIN_SCORE = 20
    MAX_MOVES = 5

    # --- Colors ---
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (60, 60, 80)
    COLOR_TEXT = (220, 220, 240)
    CRYSTAL_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 255, 80),   # Green
        3: (80, 120, 255),  # Blue
    }
    CURSOR_COLOR = (255, 255, 0)
    PARTICLE_COLOR = (255, 255, 180)

    # --- Isometric Projection ---
    ORIGIN_X = SCREEN_WIDTH // 2
    ORIGIN_Y = 100
    TILE_ISO_WIDTH_HALF = 22
    TILE_ISO_HEIGHT_HALF = 11
    CUBE_HEIGHT = 20

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.cursor_pos = [0, 0]
        self.score = 0
        self.moves_remaining = 0
        self.game_over = False
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while True:
            self._generate_initial_grid()
            if self._is_match_possible():
                break

        self.cursor_pos = [self.GRID_HEIGHT // 2, self.GRID_WIDTH // 2]
        self.score = 0
        self.moves_remaining = self.MAX_MOVES
        self.game_over = False
        self.particles = []
        self.steps = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False

        self._move_cursor(movement)

        pushed = False
        if space_pressed:
            self._handle_push('up')
            pushed = True
        elif shift_pressed:
            self._handle_push('right')
            pushed = True

        if pushed:
            # sfx_push_crystal
            self.moves_remaining -= 1
            match_info = self._resolve_board()

            if match_info['crystals_cleared'] > 0:
                reward += match_info['crystals_cleared']
                if match_info['max_combo'] == 3:
                    reward += 10
                elif match_info['max_combo'] > 3:
                    reward += 20
                # sfx_match_success
            # else: sfx_no_match

        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx_win_game
        elif self.moves_remaining <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            # sfx_lose_game
        
        # Max steps termination
        if self.steps >= 500:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move_cursor(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_HEIGHT
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_HEIGHT
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_WIDTH
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_WIDTH

    def _handle_push(self, direction):
        r, c = self.cursor_pos
        if direction == 'up':
            self.grid[:, c] = np.roll(self.grid[:, c], -1)
        elif direction == 'right':
            self.grid[r, :] = np.roll(self.grid[r, :], 1)

    def _resolve_board(self):
        total_cleared = 0
        max_combo = 0
        chain = 0
        while True:
            matches = self._find_matches()
            if not matches:
                break
            
            chain += 1
            # sfx_chain_{chain}
            
            current_max_combo = self._get_max_combo_size(matches)
            max_combo = max(max_combo, current_max_combo)

            num_cleared = len(matches)
            total_cleared += num_cleared
            self.score += num_cleared

            self._create_particles_for_matches(matches)
            self._clear_matches(matches)
            self._apply_gravity()
            self._refill_board()
        
        return {'crystals_cleared': total_cleared, 'max_combo': max_combo}

    def _find_matches(self, grid_to_check=None):
        grid = grid_to_check if grid_to_check is not None else self.grid
        matches = set()
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = grid[r, c]
                if color == 0: continue

                if c <= self.GRID_WIDTH - 3 and grid[r, c+1] == color and grid[r, c+2] == color:
                    for i in range(3): matches.add((r, c+i))
                
                if r <= self.GRID_HEIGHT - 3 and grid[r+1, c] == color and grid[r+2, c] == color:
                    for i in range(3): matches.add((r+i, c))
        return matches

    def _get_max_combo_size(self, matches):
        if not matches: return 0
        max_size = 0
        visited = set()
        match_list = list(matches)

        for r_start, c_start in match_list:
            if (r_start, c_start) in visited: continue
            
            current_size = 0
            q = [(r_start, c_start)]
            visited.add((r_start, c_start))

            head = 0
            while head < len(q):
                r, c = q[head]; head += 1
                current_size += 1
                
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in matches and (nr, nc) not in visited:
                        visited.add((nr, nc)); q.append((nr, nc))
            
            max_size = max(max_size, current_size)
        return max_size

    def _clear_matches(self, matches):
        for r, c in matches: self.grid[r, c] = 0

    def _apply_gravity(self):
        for c in range(self.GRID_WIDTH):
            col = self.grid[:, c]
            non_empty = col[col != 0]
            num_empty = self.GRID_HEIGHT - len(non_empty)
            self.grid[:, c] = np.concatenate([np.zeros(num_empty, dtype=int), non_empty])

    def _refill_board(self):
        refill_indices = np.where(self.grid == 0)
        num_to_refill = len(refill_indices[0])
        if num_to_refill > 0:
            new_crystals = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1, size=num_to_refill)
            self.grid[refill_indices] = new_crystals

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_grid_and_crystals()
        self._update_and_draw_particles()
        self._draw_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _project(self, r, c, z=0):
        x = (c - r) * self.TILE_ISO_WIDTH_HALF
        y = (c + r) * self.TILE_ISO_HEIGHT_HALF - z * self.CUBE_HEIGHT
        return int(self.ORIGIN_X + x), int(self.ORIGIN_Y + y)

    def _draw_iso_cube(self, r, c, color_tuple):
        def darken(color, factor): return tuple(max(0, int(c * factor)) for c in color)

        z_top, z_bottom = 1, 0
        color_side1, color_side2 = darken(color_tuple, 0.7), darken(color_tuple, 0.5)
        
        p_right = [self._project(r, c + 1, z_bottom), self._project(r, c + 1, z_top), self._project(r + 1, c + 1, z_top), self._project(r + 1, c + 1, z_bottom)]
        pygame.gfxdraw.filled_polygon(self.screen, p_right, color_side1)
        
        p_left = [self._project(r + 1, c, z_bottom), self._project(r + 1, c, z_top), self._project(r + 1, c + 1, z_top), self._project(r + 1, c + 1, z_bottom)]
        pygame.gfxdraw.filled_polygon(self.screen, p_left, color_side2)

        p_top = [self._project(r, c, z_top), self._project(r, c + 1, z_top), self._project(r + 1, c + 1, z_top), self._project(r + 1, c, z_top)]
        pygame.gfxdraw.filled_polygon(self.screen, p_top, color_tuple)
        
        glow_color = tuple(min(255, c_val + 60) for c_val in color_tuple)
        p_glow = [(p[0], p[1] + 2) for p in p_top]
        p_glow[0], p_glow[1], p_glow[2], p_glow[3] = (p_top[0][0], p_top[0][1] + 4), (p_top[1][0] - 4, p_top[1][1] + 2), (p_top[2][0], p_top[2][1]), (p_top[3][0] + 4, p_top[3][1] + 2)
        pygame.gfxdraw.filled_polygon(self.screen, p_glow, glow_color)

    def _draw_grid_and_crystals(self):
        for r_draw in range(self.GRID_HEIGHT * 2):
            r = r_draw // 2
            for c in range(self.GRID_WIDTH):
                if r_draw % 2 == 0:
                    p_base = [self._project(r,c), self._project(r,c+1), self._project(r+1,c+1), self._project(r+1,c)]
                    pygame.gfxdraw.aapolygon(self.screen, p_base, self.COLOR_GRID)
                else:
                    crystal_r, crystal_c = self.GRID_HEIGHT - 1 - r, c
                    color_idx = self.grid[crystal_r, crystal_c]
                    if color_idx > 0:
                        self._draw_iso_cube(crystal_r, crystal_c, self.CRYSTAL_COLORS[color_idx])

    def _draw_cursor(self):
        r, c = self.cursor_pos
        p_cursor = [self._project(r, c, 1), self._project(r, c + 1, 1), self._project(r + 1, c + 1, 1), self._project(r + 1, c, 1)]
        
        pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
        color = tuple(int(val * (0.5 + pulse * 0.5)) for val in self.CURSOR_COLOR)

        for i in range(len(p_cursor)):
            pygame.draw.line(self.screen, color, p_cursor[i], p_cursor[(i + 1) % 4], 3)

    def _render_ui(self):
        score_text = self.font_small.render(f"CRYSTALS CLEARED", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score} / {self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15)); self.screen.blit(score_val, (20, 35))

        moves_text = self.font_small.render(f"MOVES REMAINING", True, self.COLOR_TEXT)
        moves_color = self.CURSOR_COLOR if self.moves_remaining <= 1 else self.COLOR_TEXT
        moves_val = self.font_large.render(f"{self.moves_remaining}", True, moves_color)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 15))
        self.screen.blit(moves_val, (self.SCREEN_WIDTH - moves_val.get_width() - 20, 35))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "OUT OF MOVES"
            msg_render = self.font_large.render(msg, True, self.CURSOR_COLOR)
            self.screen.blit(msg_render, (self.SCREEN_WIDTH // 2 - msg_render.get_width() // 2, self.SCREEN_HEIGHT // 2 - msg_render.get_height() // 2))

    def _create_particles_for_matches(self, matches):
        for r, c in matches:
            px, py = self._project(r, c, 0.5)
            for _ in range(10):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.uniform(0.5, 1.0)
                self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]; p['vel'][1] += 0.1
            p['life'] -= 1 / 30.0
            
            if p['life'] > 0:
                active_particles.append(p)
                alpha = int(255 * (p['life'] / p['max_life']))
                radius = int(3 * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, self.PARTICLE_COLOR + (alpha,))
        self.particles = active_particles

    def _generate_initial_grid(self):
        self.grid = self.np_random.integers(1, len(self.CRYSTAL_COLORS) + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH))

    def _is_match_possible(self):
        for c in range(self.GRID_WIDTH):
            temp_grid = self.grid.copy(); temp_grid[:, c] = np.roll(temp_grid[:, c], -1)
            if self._has_match(temp_grid): return True
        
        for r in range(self.GRID_HEIGHT):
            temp_grid = self.grid.copy(); temp_grid[r, :] = np.roll(temp_grid[r, :], 1)
            if self._has_match(temp_grid): return True
            
        return False

    def _has_match(self, grid):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                color = grid[r, c]
                if color == 0: continue
                if c <= self.GRID_WIDTH - 3 and grid[r, c+1] == color and grid[r, c+2] == color: return True
                if r <= self.GRID_HEIGHT - 3 and grid[r+1, c] == color and grid[r+2, c] == color: return True
        return False

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

    def close(self):
        pygame.quit()