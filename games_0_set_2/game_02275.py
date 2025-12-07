
# Generated: 2025-08-28T04:18:15.746882
# Source Brief: brief_02275.md
# Brief Index: 2275

        
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
        "Controls: ←→ to move, ↑↓ to rotate. Press Space for hard drop, hold Shift for soft drop."
    )

    game_description = (
        "A fast-paced, retro-styled block-dropping puzzle game. Clear lines to score points, but don't let the stack reach the top!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_POS = (
            (self.screen_size[0] - self.GRID_WIDTH * self.CELL_SIZE) // 2,
            (self.screen_size[1] - self.GRID_HEIGHT * self.CELL_SIZE) // 2,
        )
        self.MAX_STEPS = 5000
        self.WIN_SCORE = 1000

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 55)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)
        self.BLOCK_COLORS = [
            (239, 131, 84),  # I (Orange)
            (93, 173, 226),  # J (Light Blue)
            (241, 196, 15),  # O (Yellow)
            (88, 101, 242),  # L (Discord Blue)
            (78, 204, 163),  # S (Green)
            (235, 87, 87),   # Z (Red)
            (170, 140, 218), # T (Purple)
        ]

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Tetromino shapes (rotations)
        self.SHAPES = [
            # I
            [[[0,1],[1,1],[2,1],[3,1]], [[1,0],[1,1],[1,2],[1,3]]],
            # J
            [[[0,0],[0,1],[1,1],[2,1]], [[1,0],[2,0],[1,1],[1,2]], [[0,1],[1,1],[2,1],[2,2]], [[1,0],[1,1],[0,2],[1,2]]],
            # O
            [[[0,0],[0,1],[1,0],[1,1]]],
            # L
            [[[2,0],[0,1],[1,1],[2,1]], [[1,0],[1,1],[1,2],[2,2]], [[0,1],[1,1],[2,1],[0,2]], [[0,0],[1,0],[1,1],[1,2]]],
            # S
            [[[1,0],[2,0],[0,1],[1,1]], [[0,0],[0,1],[1,1],[1,2]]],
            # Z
            [[[0,0],[1,0],[1,1],[2,1]], [[1,0],[0,1],[1,1],[0,2]]],
            # T
            [[[1,0],[0,1],[1,1],[2,1]], [[1,0],[0,1],[1,1],[1,2]], [[0,1],[1,1],[2,1],[1,2]], [[1,0],[1,1],[2,1],[1,2]]],
        ]

        # State variables
        self.grid = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.current_piece = None
        self.next_piece = None
        self.drop_interval_ms = 500
        self.last_drop_time = 0
        self.last_action = np.array([0, 0, 0])
        self.consecutive_placements_without_clear = 0
        self.line_clear_effects = []
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), -1, dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.drop_interval_ms = 500
        self.last_drop_time = pygame.time.get_ticks()
        self.last_action = np.array([0, 0, 0])
        self.consecutive_placements_without_clear = 0
        self.line_clear_effects = []

        self._spawn_piece()
        self._spawn_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time passing

        movement, space_press, shift_held = action[0], action[1], action[2]
        
        # --- Handle player input ---
        # Edge-triggered actions (only on press)
        rotate_cw = movement == 1 and self.last_action[0] != 1
        rotate_ccw = movement == 2 and self.last_action[0] != 2
        hard_drop = space_press == 1 and self.last_action[1] != 1

        # Level-triggered actions (while held)
        move_left = movement == 3
        move_right = movement == 4
        soft_drop = shift_held == 1

        if rotate_cw:
            self._rotate_piece(1)
        if rotate_ccw:
            self._rotate_piece(-1)
        if move_left:
            self._move_piece(-1, 0)
        if move_right:
            self._move_piece(1, 0)
        
        self.last_action = action

        if hard_drop:
            # Sound: Hard drop
            while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'] + 1)):
                self.current_piece['y'] += 1
            reward += self._lock_piece()
            self.last_drop_time = pygame.time.get_ticks()
        else:
            # --- Handle automatic drop ---
            current_time = pygame.time.get_ticks()
            drop_interval = self.drop_interval_ms / 5 if soft_drop else self.drop_interval_ms
            if current_time - self.last_drop_time > drop_interval:
                self.last_drop_time = current_time
                if not self._move_piece(0, 1):
                    # Sound: Block lock
                    reward += self._lock_piece()

        # --- Update game state and check termination ---
        old_score_tier = self.score // 200
        self.score = max(0, self.score)
        new_score_tier = self.score // 200
        if new_score_tier > old_score_tier:
            self.drop_interval_ms *= 0.9 # 10% faster

        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        if terminated:
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            elif self.game_over:
                reward -= 100 # Lose penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_piece(self):
        self.current_piece = self.next_piece
        
        shape_idx = self.rng.integers(0, len(self.SHAPES))
        self.next_piece = {
            'shape_idx': shape_idx,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 2,
            'y': 0,
            'shape': self.SHAPES[shape_idx][0]
        }
        
        if self.current_piece and self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
            self.game_over = True
            # Sound: Game Over

    def _move_piece(self, dx, dy):
        new_x = self.current_piece['x'] + dx
        new_y = self.current_piece['y'] + dy
        if not self._check_collision(self.current_piece['shape'], (new_x, new_y)):
            self.current_piece['x'] = new_x
            self.current_piece['y'] = new_y
            return True
        return False

    def _rotate_piece(self, direction):
        rotations = self.SHAPES[self.current_piece['shape_idx']]
        num_rotations = len(rotations)
        new_rotation = (self.current_piece['rotation'] + direction) % num_rotations
        new_shape = rotations[new_rotation]

        # Wall kick logic
        for dx in [0, 1, -1, 2, -2]: # Basic wall kick checks
            if not self._check_collision(new_shape, (self.current_piece['x'] + dx, self.current_piece['y'])):
                self.current_piece['rotation'] = new_rotation
                self.current_piece['shape'] = new_shape
                self.current_piece['x'] += dx
                # Sound: Rotate
                return True
        return False

    def _lock_piece(self):
        reward = 0
        for pos in self.current_piece['shape']:
            x = self.current_piece['x'] + pos[0]
            y = self.current_piece['y'] + pos[1]
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x, y] = self.current_piece['shape_idx']

        # High placement bonus
        if self.current_piece['y'] <= 1:
            reward += 20
        
        cleared_lines = self._clear_lines()
        if cleared_lines > 0:
            # Sound: Line clear
            reward += 10 * cleared_lines * cleared_lines # Bonus for multi-line clears
            self.score += 10 * cleared_lines * cleared_lines
            self.consecutive_placements_without_clear = 0
        else:
            self.consecutive_placements_without_clear += 1
            if self.consecutive_placements_without_clear >= 10:
                reward -= 2

        self._spawn_piece()
        return reward

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y] != -1):
                lines_to_clear.append(y)

        if lines_to_clear:
            for y in lines_to_clear:
                self.line_clear_effects.append({'y': y, 'timer': 10}) # 10 frames of animation
            
            # Shift grid down
            for y in sorted(lines_to_clear):
                self.grid[:, 1:y+1] = self.grid[:, 0:y]
                self.grid[:, 0] = -1
        
        return len(lines_to_clear)

    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for pos in shape:
            x, y = off_x + pos[0], off_y + pos[1]
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True
            if self.grid[x, y] != -1:
                return True
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_block(self, surface, x, y, color, is_ghost=False):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        if is_ghost:
            pygame.draw.rect(surface, color, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(surface, dark_color, rect, border_radius=4)
            pygame.draw.rect(surface, color, rect.inflate(-4, -4), border_radius=3)
            pygame.gfxdraw.aacircle(surface, rect.left + 5, rect.top + 5, 2, light_color)


    def _render_game(self):
        # Draw grid background
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        grid_surface.fill(self.COLOR_GRID)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                pygame.draw.rect(grid_surface, self.COLOR_BG, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)
        self.screen.blit(grid_surface, self.GRID_POS)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (*self.GRID_POS, grid_surface.get_width(), grid_surface.get_height()), 2, 5)

        # Draw locked blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != -1:
                    color_idx = self.grid[x, y]
                    self._draw_block(self.screen, self.GRID_POS[0] + x * self.CELL_SIZE, self.GRID_POS[1] + y * self.CELL_SIZE, self.BLOCK_COLORS[color_idx])

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self.current_piece['y']
            while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], ghost_y + 1)):
                ghost_y += 1
            for pos in self.current_piece['shape']:
                color = self.BLOCK_COLORS[self.current_piece['shape_idx']]
                self._draw_block(self.screen, self.GRID_POS[0] + (self.current_piece['x'] + pos[0]) * self.CELL_SIZE, self.GRID_POS[1] + (ghost_y + pos[1]) * self.CELL_SIZE, color, is_ghost=True)

        # Draw falling piece
        if self.current_piece and not self.game_over:
            for pos in self.current_piece['shape']:
                color = self.BLOCK_COLORS[self.current_piece['shape_idx']]
                self._draw_block(self.screen, self.GRID_POS[0] + (self.current_piece['x'] + pos[0]) * self.CELL_SIZE, self.GRID_POS[1] + (self.current_piece['y'] + pos[1]) * self.CELL_SIZE, color)

        # Draw line clear animation
        active_effects = []
        for effect in self.line_clear_effects:
            y = effect['y']
            timer = effect['timer']
            alpha = int(255 * (timer / 10))
            flash_rect = pygame.Rect(self.GRID_POS[0], self.GRID_POS[1] + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
            flash_surface = pygame.Surface(flash_rect.size, pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, flash_rect.topleft)
            effect['timer'] -= 1
            if effect['timer'] > 0:
                active_effects.append(effect)
        self.line_clear_effects = active_effects
    
    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GRID_POS[0] + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_POS[1]))
        self.screen.blit(score_val, (self.GRID_POS[0] + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_POS[1] + 30))

        # Next piece preview
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        preview_x = self.GRID_POS[0] + self.GRID_WIDTH * self.CELL_SIZE + 20
        preview_y = self.GRID_POS[1] + 100
        self.screen.blit(next_text, (preview_x, preview_y))
        
        preview_box = pygame.Rect(preview_x, preview_y + 30, 4 * self.CELL_SIZE, 4 * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, 0, 5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, preview_box, 2, 5)

        if self.next_piece:
            shape = self.next_piece['shape']
            color = self.BLOCK_COLORS[self.next_piece['shape_idx']]
            min_x = min(p[0] for p in shape)
            max_x = max(p[0] for p in shape)
            min_y = min(p[1] for p in shape)
            max_y = max(p[1] for p in shape)
            
            offset_x = preview_box.centerx - (min_x + max_x + 1) / 2 * self.CELL_SIZE
            offset_y = preview_box.centery - (min_y + max_y + 1) / 2 * self.CELL_SIZE

            for pos in shape:
                self._draw_block(self.screen, int(offset_x + pos[0] * self.CELL_SIZE), int(offset_y + pos[1] * self.CELL_SIZE), color)
        
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.score >= self.WIN_SCORE else "GAME OVER"
            go_text = self.font_main.render(status_text, True, self.COLOR_FLASH)
            go_rect = go_text.get_rect(center=(self.screen_size[0] // 2, self.screen_size[1] // 2))
            self.screen.blit(go_text, go_rect)

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

    def close(self):
        pygame.font.quit()
        pygame.quit()