
# Generated: 2025-08-27T15:17:50.298190
# Source Brief: brief_00950.md
# Brief Index: 950

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Hold Shift for soft drop, press Space for hard drop."
    )

    game_description = (
        "A fast-paced, falling block puzzle game. Clear lines to score points, but be careful not to stack blocks to the top!"
    )

    auto_advance = True

    # --- Tetromino Shapes and Colors ---
    TETROMINO_SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 0], [0, 1, 1]],  # Z
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1], [1, 1]],  # O
    ]

    TETROMINO_COLORS = [
        (66, 135, 245),   # I: Blue
        (245, 66, 66),    # Z: Red
        (66, 245, 114),   # S: Green
        (188, 66, 245),   # T: Purple
        (245, 161, 66),   # L: Orange
        (245, 239, 66),   # J: Yellow
        (66, 218, 245),   # O: Cyan
    ]

    # --- Visual Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 20
    CELL_SIZE = 18
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2 - 50
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    COLOR_BG = (20, 20, 30)
    COLOR_GRID_BG = (30, 30, 45)
    COLOR_GRID_LINES = (50, 50, 70)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GHOST = (255, 255, 255, 50)
    
    # --- Game Constants ---
    MAX_STEPS = 3000
    WIN_CONDITION_LINES = 20
    INITIAL_FALL_SPEED = 1.0 / 30.0 # 1 block per second at 30fps
    FALL_SPEED_INCREMENT = 0.05
    SOFT_DROP_MULTIPLIER = 10

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        self.grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.current_block = None
        self.next_block = None
        self.fall_counter = 0.0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.line_clear_animation = []
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.fall_counter = 0.0
        self.line_clear_animation = []
        self.particles = []
        
        self.next_block = self._new_block()
        self._spawn_block()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = -0.01  # Small penalty for time passing
        
        if self.game_over:
            return self._get_observation(), -100, True, False, self._get_info()

        # --- Handle Action Inputs ---
        if movement == 1: self._rotate(clockwise=True) # Up
        elif movement == 2: self._rotate(clockwise=False) # Down
        elif movement == 3: self._move(-1) # Left
        elif movement == 4: self._move(1) # Right
        
        block_locked_this_step = False
        
        # --- Hard Drop ---
        if space_held:
            # Sound: Hard drop
            while not self._check_collision(self.current_block['x'], self.current_block['y'] + 1):
                self.current_block['y'] += 1
            reward += self._lock_block()
            block_locked_this_step = True
        else:
            # --- Gravity and Soft Drop ---
            current_fall_increment = self.fall_speed
            if shift_held:
                current_fall_increment *= self.SOFT_DROP_MULTIPLIER
            
            self.fall_counter += current_fall_increment
            if self.fall_counter >= 1.0:
                self.fall_counter -= 1.0
                if not self._check_collision(self.current_block['x'], self.current_block['y'] + 1):
                    self.current_block['y'] += 1
                else:
                    # Sound: Block lock
                    reward += self._lock_block()
                    block_locked_this_step = True
        
        # --- Update Game State if Block was Locked ---
        if block_locked_this_step:
            cleared_count, cleared_rows = self._clear_lines()
            if cleared_count > 0:
                # Sound: Line clear
                self.line_clear_animation = [(r, 10) for r in cleared_rows] # 10 frames of animation
                line_rewards = {1: 1, 2: 2, 3: 4, 4: 8}
                reward += line_rewards.get(cleared_count, 0)
                self.lines_cleared += cleared_count
                self.score += cleared_count * 100 * cleared_count
                self.fall_speed = self.INITIAL_FALL_SPEED + (self.lines_cleared // 5) * self.FALL_SPEED_INCREMENT
            
            self._spawn_block()

        # --- Update Animations ---
        self._update_animations()

        # --- Check Termination Conditions ---
        terminated = self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS
        if self.game_over:
            reward = -100
        elif self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward = 100
            self.score += 1000 # Win bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "lines_cleared": self.lines_cleared, "steps": self.steps}

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT))

        # Draw locked blocks
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell > 0:
                    self._draw_block(x, y, cell - 1)

        # Draw ghost piece
        if not self.game_over and self.current_block:
            ghost_y = self.current_block['y']
            while not self._check_collision(self.current_block['x'], ghost_y + 1):
                ghost_y += 1
            self._draw_piece(self.current_block, self.current_block['x'], ghost_y, ghost=True)

        # Draw current falling block
        if not self.game_over and self.current_block:
            self._draw_piece(self.current_block)

        # Draw grid lines
        for i in range(self.GRID_COLS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + i * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT))
        for i in range(self.GRID_ROWS + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, self.GRID_Y + i * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + i * self.CELL_SIZE))

        # Draw line clear animation
        for y, timer in self.line_clear_animation:
            alpha = 255 * (timer / 10)
            flash_surface = pygame.Surface((self.GRID_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE))
        
        # Draw particles
        for p in self.particles:
            p_color = (255, 255, 255, p[4])
            p_surf = pygame.Surface((p[2], p[2]), pygame.SRCALPHA)
            p_surf.fill(p_color)
            self.screen.blit(p_surf, (p[0], p[1]))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (20, 20))

        # Lines
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_WHITE)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - lines_text.get_width() - 20, 20))

        # Next Piece
        next_text = self.font_small.render("NEXT", True, self.COLOR_WHITE)
        next_box_x = self.GRID_X + self.GRID_WIDTH + 20
        self.screen.blit(next_text, (next_box_x + 35, self.GRID_Y))
        if self.next_block:
            shape = self.TETROMINO_SHAPES[self.next_block['id']]
            w, h = len(shape[0]), len(shape)
            start_x = next_box_x + (100 - w * self.CELL_SIZE) / 2
            start_y = self.GRID_Y + 30 + (80 - h * self.CELL_SIZE) / 2
            self._draw_piece(self.next_block, custom_pos=(start_x, start_y), on_grid=False)
    
    def _update_animations(self):
        # Line clear flash
        self.line_clear_animation = [(y, t - 1) for y, t in self.line_clear_animation if t > 0]
        
        # Particles
        new_particles = []
        for p in self.particles:
            p[0] += p[3][0] # x += vx
            p[1] += p[3][1] # y += vy
            p[4] -= 10 # alpha -= 10
            if p[4] > 0:
                new_particles.append(p)
        self.particles = new_particles

    def _draw_piece(self, piece, piece_x=None, piece_y=None, ghost=False, custom_pos=None, on_grid=True):
        shape = piece['shape']
        px = piece_x if piece_x is not None else piece['x']
        py = piece_y if piece_y is not None else piece['y']
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    if on_grid:
                        self._draw_block(px + x, py + y, piece['id'], ghost)
                    else: # For UI like 'next piece'
                        self._draw_block_pixel(custom_pos[0] + x * self.CELL_SIZE, custom_pos[1] + y * self.CELL_SIZE, piece['id'])

    def _draw_block(self, x, y, color_id, ghost=False):
        pixel_x = self.GRID_X + x * self.CELL_SIZE
        pixel_y = self.GRID_Y + y * self.CELL_SIZE
        self._draw_block_pixel(pixel_x, pixel_y, color_id, ghost)

    def _draw_block_pixel(self, x, y, color_id, ghost=False):
        color = self.TETROMINO_COLORS[color_id]
        if ghost:
            pygame.gfxdraw.box(self.screen, (int(x), int(y), self.CELL_SIZE, self.CELL_SIZE), self.COLOR_GHOST)
            pygame.gfxdraw.rectangle(self.screen, (int(x), int(y), self.CELL_SIZE, self.CELL_SIZE), (255, 255, 255, 100))
        else:
            main_color = tuple(min(255, c + 30) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            
            pygame.draw.rect(self.screen, dark_color, (x, y, self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(self.screen, main_color, (x+1, y+1, self.CELL_SIZE-2, self.CELL_SIZE-2))

    def _new_block(self):
        block_id = self.np_random.integers(0, len(self.TETROMINO_SHAPES))
        shape = self.TETROMINO_SHAPES[block_id]
        return {
            'id': block_id,
            'shape': shape,
            'x': self.GRID_COLS // 2 - len(shape[0]) // 2,
            'y': 0
        }

    def _spawn_block(self):
        self.current_block = self.next_block
        self.next_block = self._new_block()
        if self._check_collision(self.current_block['x'], self.current_block['y']):
            self.game_over = True

    def _check_collision(self, piece_x, piece_y, shape=None):
        shape = shape if shape is not None else self.current_block['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = piece_x + x, piece_y + y
                    if not (0 <= grid_x < self.GRID_COLS and 0 <= grid_y < self.GRID_ROWS):
                        return True  # Out of bounds
                    if self.grid[grid_y][grid_x] != 0:
                        return True  # Collision with another block
        return False

    def _rotate(self, clockwise=True):
        shape = self.current_block['shape']
        if self.current_block['id'] == 6: return # 'O' block doesn't rotate

        rotated = list(zip(*shape[::-1])) if clockwise else list(zip(*shape))[::-1]
        
        # Wall kick
        if not self._check_collision(self.current_block['x'], self.current_block['y'], rotated):
            self.current_block['shape'] = rotated
        elif not self._check_collision(self.current_block['x'] + 1, self.current_block['y'], rotated):
            self.current_block['x'] += 1
            self.current_block['shape'] = rotated
        elif not self._check_collision(self.current_block['x'] - 1, self.current_block['y'], rotated):
            self.current_block['x'] -= 1
            self.current_block['shape'] = rotated

    def _move(self, dx):
        if not self._check_collision(self.current_block['x'] + dx, self.current_block['y']):
            self.current_block['x'] += dx

    def _lock_block(self):
        risk_reward = self._calculate_risk_reward()
        shape = self.current_block['shape']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = self.current_block['x'] + x
                    grid_y = self.current_block['y'] + y
                    if 0 <= grid_y < self.GRID_ROWS and 0 <= grid_x < self.GRID_COLS:
                        self.grid[grid_y][grid_x] = self.current_block['id'] + 1
        self.current_block = None
        return risk_reward

    def _calculate_risk_reward(self):
        shape = self.current_block['shape']
        base_y = self.current_block['y']
        
        min_y_in_piece = self.GRID_ROWS
        occupied_cols = set()
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    min_y_in_piece = min(min_y_in_piece, base_y + y)
                    occupied_cols.add(self.current_block['x'] + x)
        
        is_risky = False
        is_safe = True
        
        for col in occupied_cols:
            if 0 <= col < self.GRID_COLS:
                empty_below = 0
                for row in range(min_y_in_piece + 1, self.GRID_ROWS):
                    if self.grid[row][col] == 0:
                        empty_below += 1
                
                if empty_below < 3:
                    is_risky = True
                if empty_below <= 6:
                    is_safe = False

        if is_risky: return 1.0
        if is_safe: return -0.2
        return 0.0

    def _clear_lines(self):
        lines_to_clear = []
        for y, row in enumerate(self.grid):
            if all(cell > 0 for cell in row):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0, []

        for y in lines_to_clear:
            self.grid.pop(y)
            self.grid.insert(0, [0 for _ in range(self.GRID_COLS)])
            # Spawn particles
            for _ in range(30):
                px = self.GRID_X + self.np_random.uniform(0, self.GRID_WIDTH)
                py = self.GRID_Y + y * self.CELL_SIZE + self.np_random.uniform(-self.CELL_SIZE//2, self.CELL_SIZE//2)
                vx = self.np_random.uniform(-1.5, 1.5)
                vy = self.np_random.uniform(-1.5, 1.5)
                size = self.np_random.uniform(2, 5)
                alpha = self.np_random.integers(150, 255)
                self.particles.append([px, py, size, (vx, vy), alpha])

        return len(lines_to_clear), lines_to_clear

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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")