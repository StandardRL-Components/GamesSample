
# Generated: 2025-08-27T17:17:31.606184
# Source Brief: brief_01480.md
# Brief Index: 1480

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Press Space for hard drop."
    )

    game_description = (
        "A classic block-stacking puzzle. Strategically place falling shapes to clear lines and achieve a high score."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAYFIELD_WIDTH = 10
    PLAYFIELD_HEIGHT = 20
    BLOCK_SIZE = 18
    PLAYFIELD_PIXEL_WIDTH = PLAYFIELD_WIDTH * BLOCK_SIZE
    PLAYFIELD_PIXEL_HEIGHT = PLAYFIELD_HEIGHT * BLOCK_SIZE

    SIDE_PANEL_WIDTH = 200
    PLAYFIELD_X_OFFSET = (SCREEN_WIDTH - PLAYFIELD_PIXEL_WIDTH) // 2
    PLAYFIELD_Y_OFFSET = (SCREEN_HEIGHT - PLAYFIELD_PIXEL_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 215, 0)
    COLOR_WHITE = (255, 255, 255)
    
    # Tetromino shapes and their colors
    TETROMINOES = {
        'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
        'O': [[[1, 1], [1, 1]]],
        'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
        'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]],
        'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
        'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
        'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]]
    }
    TETROMINO_COLORS = [
        (0, 0, 0),        # 0: Empty
        (0, 240, 240),    # 1: I (Cyan)
        (240, 240, 0),    # 2: O (Yellow)
        (160, 0, 240),    # 3: T (Purple)
        (240, 160, 0),    # 4: L (Orange)
        (0, 0, 240),      # 5: J (Blue)
        (0, 240, 0),      # 6: S (Green)
        (240, 0, 0),      # 7: Z (Red)
    ]
    
    # Game parameters
    MAX_STEPS = 3000
    WIN_CONDITION_LINES = 10
    FALL_SPEED_NORMAL = 15  # Ticks per grid cell drop
    FALL_SPEED_SOFT_DROP = 3
    MOVE_COOLDOWN = 4
    ROTATE_COOLDOWN = 6

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.block_queue = []
        self._fill_block_queue()
        # Initial difficulty rule: First two blocks are 'I' shapes
        self.block_queue.insert(0, 'I')
        self.block_queue.insert(0, 'I')
        
        self.current_block = None
        self.next_block_shape = self.block_queue.pop(0)
        self._spawn_new_block()

        self.fall_timer = 0
        self.move_timer = 0
        self.rotate_timer = 0
        self.space_was_held = False
        
        self.line_clear_animation = [] # list of (row_index, timer)
        self.particles = []

        return self._get_observation(), self._get_info()

    def _fill_block_queue(self):
        bag = list(self.TETROMINOES.keys())
        self.np_random.shuffle(bag)
        self.block_queue.extend(bag)

    def _spawn_new_block(self):
        if not self.block_queue:
            self._fill_block_queue()
        
        self.current_block_shape = self.next_block_shape
        self.next_block_shape = self.block_queue.pop(0)
        
        shape_data = self.TETROMINOES[self.current_block_shape][0]
        color_idx = list(self.TETROMINOES.keys()).index(self.current_block_shape) + 1
        
        self.current_block = {
            'shape_id': self.current_block_shape,
            'rotation': 0,
            'row': 0,
            'col': (self.PLAYFIELD_WIDTH - len(shape_data[0])) // 2,
            'color_idx': color_idx,
        }
        
        # Check for game over
        if not self._is_valid_position():
            self.game_over = True
            self.current_block = None

    def _is_valid_position(self, offset_row=0, offset_col=0, rotation=None):
        if self.current_block is None:
            return False
            
        shape_id = self.current_block['shape_id']
        rot_idx = rotation if rotation is not None else self.current_block['rotation']
        shape = self.TETROMINOES[shape_id][rot_idx % len(self.TETROMINOES[shape_id])]
        
        row_pos = self.current_block['row'] + offset_row
        col_pos = self.current_block['col'] + offset_col

        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    grid_r, grid_c = row_pos + r, col_pos + c
                    if not (0 <= grid_c < self.PLAYFIELD_WIDTH and 0 <= grid_r < self.PLAYFIELD_HEIGHT):
                        return False # Out of bounds
                    if self.grid[grid_r, grid_c] != 0:
                        return False # Collision with placed block
        return True

    def _move(self, dx):
        if self._is_valid_position(offset_col=dx):
            self.current_block['col'] += dx
            # Sfx: move.wav
            return True
        return False

    def _rotate(self):
        if self.current_block is None: return False
        
        current_rotation = self.current_block['rotation']
        next_rotation = (current_rotation + 1) % len(self.TETROMINOES[self.current_block['shape_id']])
        
        # Try to rotate
        if self._is_valid_position(rotation=next_rotation):
            self.current_block['rotation'] = next_rotation
            # Sfx: rotate.wav
            return True

        # Wall kick logic
        for kick_offset in [-1, 1, -2, 2]:
            if self._is_valid_position(offset_col=kick_offset, rotation=next_rotation):
                self.current_block['col'] += kick_offset
                self.current_block['rotation'] = next_rotation
                # Sfx: rotate.wav
                return True
        return False

    def _place_block(self):
        if self.current_block is None: return
        
        shape_id = self.current_block['shape_id']
        rot_idx = self.current_block['rotation']
        shape = self.TETROMINOES[shape_id][rot_idx % len(self.TETROMINOES[shape_id])]
        
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    grid_r = self.current_block['row'] + r
                    grid_c = self.current_block['col'] + c
                    if 0 <= grid_r < self.PLAYFIELD_HEIGHT and 0 <= grid_c < self.PLAYFIELD_WIDTH:
                        self.grid[grid_r, grid_c] = self.current_block['color_idx']
        
        self.current_block = None
        # Sfx: place_block.wav
    
    def _hard_drop(self):
        if self.current_block is None: return 0
        
        # Find landing position
        drop_rows = 0
        while self._is_valid_position(offset_row=drop_rows + 1):
            drop_rows += 1
        
        self.current_block['row'] += drop_rows
        
        # Reward calculation before placement
        reward, _ = self._calculate_placement_reward()
        
        self._place_block()
        # Sfx: hard_drop.wav
        return reward

    def _count_holes(self, grid):
        holes = 0
        for c in range(self.PLAYFIELD_WIDTH):
            found_block = False
            for r in range(self.PLAYFIELD_HEIGHT):
                if grid[r, c] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def _calculate_placement_reward(self):
        if self.current_block is None: return 0, []

        temp_grid = self.grid.copy()
        shape_id = self.current_block['shape_id']
        rot_idx = self.current_block['rotation']
        shape = self.TETROMINOES[shape_id][rot_idx % len(self.TETROMINOES[shape_id])]
        
        placed_rows = set()
        for r, row_data in enumerate(shape):
            for c, cell in enumerate(row_data):
                if cell:
                    grid_r = self.current_block['row'] + r
                    grid_c = self.current_block['col'] + c
                    if 0 <= grid_r < self.PLAYFIELD_HEIGHT:
                         temp_grid[grid_r, grid_c] = self.current_block['color_idx']
                         placed_rows.add(grid_r)
        
        reward = 0
        for r_idx in placed_rows:
            if not np.all(temp_grid[r_idx, :] != 0): # if row is not full
                reward += 0.1
        
        holes_before = self._count_holes(self.grid)
        holes_after = self._count_holes(temp_grid)
        new_holes = holes_after - holes_before
        reward -= max(0, new_holes) * 0.2
        
        return reward, temp_grid

    def _clear_lines(self):
        full_lines = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                full_lines.append(r)
        
        if not full_lines:
            return 0
        
        for r in full_lines:
            self.line_clear_animation.append([r, 10]) # 10 frames of animation
            # Sfx: line_clear_start.wav
            
        for r in full_lines:
            for c in range(self.PLAYFIELD_WIDTH):
                px = self.PLAYFIELD_X_OFFSET + c * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
                py = self.PLAYFIELD_Y_OFFSET + r * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
                color = self.TETROMINO_COLORS[self.grid[r, c]]
                for _ in range(3):
                    self.particles.append(Particle(px, py, color, self.np_random))
        
        for r in sorted(full_lines, reverse=True):
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        
        self.lines_cleared += len(full_lines)
        return len(full_lines)

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        reward = 0
        terminated = False
        
        if self.game_over:
            reward = -100
            terminated = True
        else:
            if self.line_clear_animation:
                self.line_clear_animation = [[r, t-1] for r, t in self.line_clear_animation if t > 1]
            
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self.move_timer = max(0, self.move_timer - 1)
            self.rotate_timer = max(0, self.rotate_timer - 1)

            if self.current_block:
                if movement == 3 and self.move_timer == 0: # Left
                    if self._move(-1): self.move_timer = self.MOVE_COOLDOWN
                elif movement == 4 and self.move_timer == 0: # Right
                    if self._move(1): self.move_timer = self.MOVE_COOLDOWN
                elif movement == 1 and self.rotate_timer == 0: # Up (Rotate)
                    if self._rotate(): self.rotate_timer = self.ROTATE_COOLDOWN

                if space_held and not self.space_was_held:
                    reward += self._hard_drop()
                    lines = self._clear_lines()
                    if lines > 0:
                        reward += {1: 1, 2: 3, 3: 7, 4: 15}.get(lines, 0)
                        self.score += {1: 10, 2: 30, 3: 60, 4: 100}.get(lines, 0)
                    
                    self._spawn_new_block()
                
            self.space_was_held = space_held

            if self.current_block:
                is_soft_dropping = (movement == 2)
                current_fall_speed = self.FALL_SPEED_SOFT_DROP if is_soft_dropping else self.FALL_SPEED_NORMAL
                
                self.fall_timer += 1
                if self.fall_timer >= current_fall_speed:
                    self.fall_timer = 0
                    if self._is_valid_position(offset_row=1):
                        self.current_block['row'] += 1
                    else: # Block landed
                        # Sfx: land.wav
                        placement_reward, _ = self._calculate_placement_reward()
                        reward += placement_reward
                        self._place_block()
                        
                        lines = self._clear_lines()
                        if lines > 0:
                            reward += {1: 1, 2: 3, 3: 7, 4: 15}.get(lines, 0)
                            self.score += {1: 10, 2: 30, 3: 60, 4: 100}.get(lines, 0)
                        
                        self._spawn_new_block()
        
        self.steps += 1
        
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if self.lines_cleared < self.WIN_CONDITION_LINES:
                 self.game_over = True

        if self.game_over and not terminated:
            reward = -100
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_playfield()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_block(self, surface, r, c, color_idx, x_offset, y_offset, alpha=255):
        color = self.TETROMINO_COLORS[color_idx]
        darker_color = tuple(max(0, val - 50) for val in color)
        
        px, py = x_offset + c * self.BLOCK_SIZE, y_offset + r * self.BLOCK_SIZE
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        if alpha < 255:
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill((*color, alpha))
            surface.blit(s, rect.topleft)
        else:
            pygame.draw.rect(surface, darker_color, rect)
            inner_rect = pygame.Rect(px + 2, py + 2, self.BLOCK_SIZE - 4, self.BLOCK_SIZE - 4)
            pygame.draw.rect(surface, color, inner_rect)
            
    def _render_playfield(self):
        for r in range(self.PLAYFIELD_HEIGHT + 1):
            y = self.PLAYFIELD_Y_OFFSET + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAYFIELD_X_OFFSET, y), (self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_PIXEL_WIDTH, y))
        for c in range(self.PLAYFIELD_WIDTH + 1):
            x = self.PLAYFIELD_X_OFFSET + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.PLAYFIELD_Y_OFFSET), (x, self.PLAYFIELD_Y_OFFSET + self.PLAYFIELD_PIXEL_HEIGHT))
            
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.grid[r, c] != 0:
                    self._render_block(self.screen, r, c, self.grid[r, c], self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET)

        if self.current_block:
            drop_rows = 0
            while self._is_valid_position(offset_row=drop_rows + 1):
                drop_rows += 1
            
            shape_id = self.current_block['shape_id']
            rot_idx = self.current_block['rotation']
            shape = self.TETROMINOES[shape_id][rot_idx % len(self.TETROMINOES[shape_id])]
            ghost_r = self.current_block['row'] + drop_rows
            ghost_c = self.current_block['col']
            
            for r_off, row_data in enumerate(shape):
                for c_off, cell in enumerate(row_data):
                    if cell:
                        self._render_block(self.screen, ghost_r + r_off, ghost_c + c_off, self.current_block['color_idx'], self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET, alpha=60)
        
        if self.current_block:
            shape_id = self.current_block['shape_id']
            rot_idx = self.current_block['rotation']
            shape = self.TETROMINOES[shape_id][rot_idx % len(self.TETROMINOES[shape_id])]
            
            for r_off, row_data in enumerate(shape):
                for c_off, cell in enumerate(row_data):
                    if cell:
                        self._render_block(self.screen, self.current_block['row'] + r_off, self.current_block['col'] + c_off, self.current_block['color_idx'], self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET)

        for r, t in self.line_clear_animation:
            alpha = 255 * (t / 10)
            rect = pygame.Rect(self.PLAYFIELD_X_OFFSET, self.PLAYFIELD_Y_OFFSET + r * self.BLOCK_SIZE, self.PLAYFIELD_PIXEL_WIDTH, self.BLOCK_SIZE)
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_WHITE, alpha))
            self.screen.blit(s, rect.topleft)
            
    def _render_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            p.draw(self.screen)
            
    def _render_ui(self):
        left_panel_x = 30
        
        score_title = self.font_medium.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_title, (left_panel_x, 40))
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_val, (left_panel_x, 70))
        
        next_title = self.font_medium.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_title, (left_panel_x, 140))
        
        next_shape = self.TETROMINOES[self.next_block_shape][0]
        color_idx = list(self.TETROMINOES.keys()).index(self.next_block_shape) + 1
        shape_w = len(next_shape[0])
        shape_h = len(next_shape)
        
        for r_off, row_data in enumerate(next_shape):
            for c_off, cell in enumerate(row_data):
                if cell:
                    px = left_panel_x + (c_off * self.BLOCK_SIZE) + (4-shape_w) * self.BLOCK_SIZE/2
                    py = 180 + (r_off * self.BLOCK_SIZE) + (4-shape_h) * self.BLOCK_SIZE/2
                    self._render_block(self.screen, 0, 0, color_idx, int(px), int(py))

        right_panel_x = self.PLAYFIELD_X_OFFSET + self.PLAYFIELD_PIXEL_WIDTH + 30
        
        lines_title = self.font_medium.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_title, (right_panel_x, 40))
        lines_val = self.font_large.render(f"{self.lines_cleared} / {self.WIN_CONDITION_LINES}", True, self.COLOR_WHITE)
        self.screen.blit(lines_val, (right_panel_x, 70))
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "GAME OVER"
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                message = "YOU WIN!"
                
            text = self.font_large.render(message, True, self.COLOR_WHITE)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()
        
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

class Particle:
    def __init__(self, x, y, color, rng):
        self.x = x
        self.y = y
        self.vx = rng.uniform(-2.5, 2.5)
        self.vy = rng.uniform(-4, -1)
        self.color = color
        self.max_lifespan = rng.integers(20, 40)
        self.lifespan = self.max_lifespan
        self.size = rng.integers(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15
        self.lifespan -= 1

    def draw(self, surface):
        if self.lifespan > 0:
            life_ratio = self.lifespan / self.max_lifespan
            current_size = int(self.size * life_ratio)
            if current_size > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), current_size, (*self.color, int(255 * life_ratio)))