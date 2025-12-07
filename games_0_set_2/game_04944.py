
# Generated: 2025-08-28T03:31:31.005062
# Source Brief: brief_04944.md
# Brief Index: 4944

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate CW, Shift to rotate CCW. "
        "↓ to soft drop, Space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric block-stacking puzzle game. Clear lines to score points. "
        "Win by clearing 50 lines, lose if the blocks reach the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.MAX_STEPS = 3000
        self.WIN_LINES = 50

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
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        
        # Tetrominoes (S, Z, I, O, J, L, T)
        self.BLOCK_COLORS = [
            (0, 0, 0),         # 0: Empty
            (50, 205, 50),     # 1: S (Green)
            (220, 20, 60),     # 2: Z (Red)
            (0, 191, 255),     # 3: I (Cyan)
            (255, 215, 0),     # 4: O (Yellow)
            (0, 0, 205),       # 5: J (Blue)
            (255, 140, 0),     # 6: L (Orange)
            (148, 0, 211),     # 7: T (Purple)
        ]
        
        self.BLOCK_SHAPES = [
            [[[0,1,1],[1,1,0],[0,0,0]], [[1,0,0],[1,1,0],[0,1,0]]],
            [[[1,1,0],[0,1,1],[0,0,0]], [[0,1,0],[1,1,0],[1,0,0]]],
            [[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]], [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]],
            [[[1,1],[1,1]]],
            [[[1,0,0],[1,1,1],[0,0,0]], [[0,1,1],[0,1,0],[0,1,0]], [[0,0,0],[1,1,1],[0,0,1]], [[0,1,0],[0,1,0],[1,1,0]]],
            [[[0,0,1],[1,1,1],[0,0,0]], [[0,1,0],[0,1,0],[0,1,1]], [[0,0,0],[1,1,1],[1,0,0]], [[1,1,0],[0,1,0],[0,1,0]]],
            [[[0,1,0],[1,1,1],[0,0,0]], [[0,1,0],[0,1,1],[0,1,0]], [[0,0,0],[1,1,1],[0,1,0]], [[0,1,0],[1,1,0],[0,1,0]]],
        ]
        
        # Isometric rendering constants
        self.TILE_WIDTH, self.TILE_HEIGHT, self.TILE_DEPTH = 20, 10, 10
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 60

        # State variables (initialized in reset)
        self.playfield = None
        self.current_block = None
        self.next_block_id = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.reward = 0
        self.particles = []
        self.line_clear_frames = 0
        self.cleared_rows = []
        self.fall_counter = 0
        self.fall_speed = 20 # frames per grid cell drop
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.playfield = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.reward = 0
        self.particles = []
        self.line_clear_frames = 0
        self.cleared_rows = []
        self.fall_counter = 0

        self.next_block_id = self.np_random.integers(0, 7)
        self._spawn_new_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        self.reward = -0.02

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.line_clear_frames > 0:
            self.line_clear_frames -= 1
            if self.line_clear_frames == 0:
                self._perform_line_clear()
        elif not self.game_over:
            if space_held and self.current_block:
                self._hard_drop()
            else:
                if self.current_block:
                    if movement == 1: self._rotate_block(1)
                    if shift_held: self._rotate_block(-1)
                    if movement == 3: self._move_block(-1)
                    elif movement == 4: self._move_block(1)

                soft_drop = (movement == 2)
                self.fall_counter += 2 if soft_drop else 1
                if soft_drop and self.current_block: self.reward += 0.01

                if self.fall_counter >= self.fall_speed:
                    self.fall_counter = 0
                    if self.current_block:
                        new_y = self.current_block['y'] + 1
                        if not self._check_collision(self.current_block['shape'], (self.current_block['x'], new_y)):
                            self.current_block['y'] = new_y
                        else:
                            self._lock_block()
        
        self._update_particles()
        
        terminated = self.game_over or self.lines_cleared >= self.WIN_LINES or self.steps >= self.MAX_STEPS
        if terminated:
            if self.lines_cleared >= self.WIN_LINES: self.reward += 100
            elif self.game_over: self.reward -= 10
            
        return self._get_observation(), self.reward, terminated, False, self._get_info()

    def _spawn_new_block(self):
        self.current_block_id = self.next_block_id
        self.next_block_id = self.np_random.integers(0, 7)
        shape_list = self.BLOCK_SHAPES[self.current_block_id]
        
        self.current_block = {
            'id': self.current_block_id + 1,
            'shapes': shape_list, 'shape_idx': 0, 'shape': shape_list[0],
            'x': self.GRID_WIDTH // 2 - len(shape_list[0][0]) // 2, 'y': 0,
        }
        if self._check_collision(self.current_block['shape'], (self.current_block['x'], self.current_block['y'])):
            self.game_over = True
            self.current_block = None

    def _check_collision(self, shape, pos):
        x, y = pos
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = x + c, y + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT): return True
                    if self.playfield[grid_y, grid_x] != 0: return True
        return False

    def _rotate_block(self, direction):
        b = self.current_block
        new_idx = (b['shape_idx'] + direction) % len(b['shapes'])
        new_shape = b['shapes'][new_idx]
        for offset in [0, 1, -1, 2, -2]:
            if not self._check_collision(new_shape, (b['x'] + offset, b['y'])):
                b['shape_idx'], b['shape'], b['x'] = new_idx, new_shape, b['x'] + offset
                return True
        return False

    def _move_block(self, dx):
        b = self.current_block
        if not self._check_collision(b['shape'], (b['x'] + dx, b['y'])):
            b['x'] += dx

    def _hard_drop(self):
        ghost_y = self._get_ghost_position()
        drop_distance = ghost_y - self.current_block['y']
        self.reward += 0.1 * drop_distance
        self.current_block['y'] = ghost_y
        self._lock_block()

    def _get_ghost_position(self):
        y = self.current_block['y']
        while not self._check_collision(self.current_block['shape'], (self.current_block['x'], y + 1)):
            y += 1
        return y

    def _lock_block(self):
        self._calculate_placement_reward()
        b = self.current_block
        for r, row in enumerate(b['shape']):
            for c, cell in enumerate(row):
                if cell:
                    gy, gx = b['y'] + r, b['x'] + c
                    if 0 <= gy < self.GRID_HEIGHT and 0 <= gx < self.GRID_WIDTH:
                        self.playfield[gy, gx] = b['id']
        
        lines_to_clear = [r for r in range(self.GRID_HEIGHT) if np.all(self.playfield[r, :] != 0)]
        
        if lines_to_clear:
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            rewards = {1: 1, 2: 2, 3: 4, 4: 8}
            self.reward += rewards.get(num_cleared, 8)
            self.score += (100 * num_cleared) * num_cleared
            self.cleared_rows, self.line_clear_frames = lines_to_clear, 10
            # sfx: line_clear_explode
            for r in lines_to_clear:
                for c in range(self.GRID_WIDTH): self._create_particles(c, r)
        else:
            self._spawn_new_block()
        self.current_block = None

    def _perform_line_clear(self):
        rows_to_move = self.playfield.tolist()
        for r in sorted(self.cleared_rows, reverse=True): del rows_to_move[r]
        for _ in self.cleared_rows: rows_to_move.insert(0, [0] * self.GRID_WIDTH)
        self.playfield = np.array(rows_to_move, dtype=int)
        self.cleared_rows = []
        self._spawn_new_block()

    def _calculate_placement_reward(self):
        b = self.current_block
        shape, x, y = b['shape'], b['x'], self._get_ghost_position()
        support_cells, bottom_cells = 0, 0
        for r in range(len(shape) - 1, -1, -1):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    if r == len(shape) - 1 or shape[r + 1][c] == 0:
                        bottom_cells += 1
                        gy, gx = y + r, x + c
                        if gy + 1 >= self.GRID_HEIGHT or self.playfield[gy + 1, gx] != 0:
                            support_cells += 1
        if bottom_cells > 0:
            if support_cells == 1: self.reward -= 0.5
            elif support_cells >= 2: self.reward += 0.5

    def _to_iso(self, x, y, z=0):
        iso_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        iso_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2 - z * self.TILE_DEPTH
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, color, grid_x, grid_y, z_offset=0):
        side_color = tuple(max(0, c - 50) for c in color)
        dark_side_color = tuple(max(0, c - 80) for c in color)
        points = [(grid_x, grid_y, z_offset), (grid_x + 1, grid_y, z_offset), (grid_x + 1, grid_y + 1, z_offset), (grid_x, grid_y + 1, z_offset),
                  (grid_x, grid_y, z_offset + 1), (grid_x + 1, grid_y, z_offset + 1), (grid_x + 1, grid_y + 1, z_offset + 1), (grid_x, grid_y + 1, z_offset + 1)]
        iso_points = [self._to_iso(p[0], p[1], p[2]) for p in points]
        pygame.gfxdraw.filled_polygon(surface, [iso_points[4], iso_points[5], iso_points[6], iso_points[7]], color)
        pygame.gfxdraw.filled_polygon(surface, [iso_points[0], iso_points[3], iso_points[7], iso_points[4]], side_color)
        pygame.gfxdraw.filled_polygon(surface, [iso_points[0], iso_points[1], iso_points[5], iso_points[4]], dark_side_color)
        outline_color = (0,0,0,100)
        pygame.gfxdraw.aapolygon(surface, [iso_points[4], iso_points[5], iso_points[6], iso_points[7]], outline_color)
        pygame.gfxdraw.aapolygon(surface, [iso_points[0], iso_points[3], iso_points[7], iso_points[4]], outline_color)
        pygame.gfxdraw.aapolygon(surface, [iso_points[0], iso_points[1], iso_points[5], iso_points[4]], outline_color)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_HEIGHT + 1): pygame.draw.aaline(self.screen, self.COLOR_GRID, self._to_iso(0, r), self._to_iso(self.GRID_WIDTH, r))
        for c in range(self.GRID_WIDTH + 1): pygame.draw.aaline(self.screen, self.COLOR_GRID, self._to_iso(c, 0), self._to_iso(c, self.GRID_HEIGHT))

        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                cell_id = self.playfield[r, c]
                if cell_id != 0:
                    if self.line_clear_frames > 0 and r in self.cleared_rows:
                        flash_color = self.COLOR_WHITE if (self.line_clear_frames // 2) % 2 == 0 else self.BLOCK_COLORS[cell_id]
                        self._draw_iso_cube(self.screen, flash_color, c, r)
                    else:
                        self._draw_iso_cube(self.screen, self.BLOCK_COLORS[cell_id], c, r)
        
        if self.current_block:
            b = self.current_block
            ghost_y = self._get_ghost_position()
            ghost_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            for r_off, row in enumerate(b['shape']):
                for c_off, cell in enumerate(row):
                    if cell: self._draw_iso_cube(ghost_surface, (200, 200, 220), b['x'] + c_off, ghost_y + r_off)
            ghost_surface.set_alpha(80)
            self.screen.blit(ghost_surface, (0, 0))
            for r_off, row in enumerate(b['shape']):
                for c_off, cell in enumerate(row):
                    if cell: self._draw_iso_cube(self.screen, self.BLOCK_COLORS[b['id']], b['x'] + c_off, b['y'] + r_off)
        self._draw_particles()

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        lines_text = self.font_ui.render(f"Lines: {self.lines_cleared}/{self.WIN_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.WIDTH - lines_text.get_width() - 10, 10))
        next_text = self.font_ui.render("Next:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 120, self.HEIGHT - 120))
        
        next_shape, next_color = self.BLOCK_SHAPES[self.next_block_id][0], self.BLOCK_COLORS[self.next_block_id + 1]
        px, py = self.WIDTH - 80, self.HEIGHT - 80
        for r_off, row in enumerate(next_shape):
            for c_off, cell in enumerate(row):
                if cell:
                    iso_x, iso_y = px + (c_off - r_off) * 10, py + (c_off + r_off) * 5
                    pygame.draw.polygon(self.screen, next_color, [(iso_x, iso_y - 5), (iso_x + 10, iso_y), (iso_x, iso_y + 5), (iso_x - 10, iso_y)])
        
        msg = "GAME OVER" if self.game_over else "YOU WIN!" if self.lines_cleared >= self.WIN_LINES else None
        if msg:
            text_surf = self.font_game_over.render(msg, True, self.COLOR_WHITE)
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2)))

    def _create_particles(self, grid_x, grid_y):
        iso_x, iso_y = self._to_iso(grid_x, grid_y, 1)
        iso_x += self.TILE_WIDTH / 2
        color = self.BLOCK_COLORS[self.playfield[grid_y, grid_x]]
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({'pos': [iso_x, iso_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            size = int(3 * (p['life'] / p['max_life']))
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, p['color'] + (alpha,), (size, size), size)
                self.screen.blit(temp_surf, (pos[0]-size, pos[1]-size))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared}

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and obs.dtype == np.uint8 and isinstance(info, dict)
        action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3) and obs.dtype == np.uint8
        assert isinstance(reward, (int, float)) and isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")