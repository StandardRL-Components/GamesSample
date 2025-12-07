
# Generated: 2025-08-27T21:45:57.968586
# Source Brief: brief_01539.md
# Brief Index: 1539

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import collections
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ for soft drop. "
        "Hold Shift to rotate counter-clockwise. Press Space for hard drop."
    )

    game_description = (
        "A fast-paced, grid-based puzzle game. Manipulate falling shapes to "
        "complete horizontal lines. Clear 10 lines to win, but don't let the stack reach the top!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 10

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_UI_BG = (30, 30, 50)
        self.SHAPE_COLORS = [
            (0, 240, 240),  # I (Cyan)
            (240, 240, 0),  # O (Yellow)
            (160, 0, 240),  # T (Purple)
            (0, 0, 240),    # J (Blue)
            (240, 160, 0),  # L (Orange)
            (0, 240, 0),    # S (Green)
            (240, 0, 0),    # Z (Red)
        ]

        # --- Tetromino Shapes ---
        self.TETROMINOES = {
            0: [[1, 1, 1, 1]],  # I
            1: [[1, 1], [1, 1]],  # O
            2: [[0, 1, 0], [1, 1, 1]],  # T
            3: [[1, 0, 0], [1, 1, 1]],  # J
            4: [[0, 0, 1], [1, 1, 1]],  # L
            5: [[0, 1, 1], [1, 1, 0]],  # S
            6: [[1, 1, 0], [0, 1, 1]],  # Z
        }
        
        # --- Input Handling Constants ---
        self.DAS_DELAY = 8  # Delayed Auto-Shift delay in frames
        self.DAS_RATE = 2   # DAS repeat rate in frames

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State Variables ---
        # These are initialized in reset()
        self.grid = None
        self.current_shape_id = None
        self.current_shape = None
        self.current_pos = None
        self.next_shape_id = None
        self.score = None
        self.lines_cleared = None
        self.steps = None
        self.game_over = None
        self.drop_counter = None
        self.drop_speed_frames = None
        self.particles = None
        self.key_hold_timers = None
        self.np_random = None

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.drop_speed_frames = self.FPS  # Start at 1 cell per second
        self.drop_counter = 0
        self.particles = []
        self.key_hold_timers = collections.defaultdict(int)

        self._spawn_shape(initial=True)
        self._spawn_shape()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed
        self.steps += 1

        if not self.game_over:
            reward_from_action = self._handle_input(action)
            reward += reward_from_action
            
            if not self.game_over: # Input might have caused game over
                self._update_game_state()

        terminated = self._check_termination()
        if terminated and not self.game_over: # Win condition met
            reward += 100
            self.game_over = True
        elif self.game_over: # Lose condition was met
            reward += -100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Hard Drop (Space) ---
        if space_held:
            # Find landing spot
            while not self._check_collision(self.current_shape, (self.current_pos[0], self.current_pos[1] + 1)):
                self.current_pos[1] += 1
            # Lock and process
            reward += self._lock_shape()
            if not self.game_over:
                self._spawn_shape()
            return reward # Hard drop action ends the player's turn for this frame

        # --- DAS Input Handling ---
        actions_map = {1: 'up', 2: 'down', 3: 'left', 4: 'right'}
        active_move = actions_map.get(movement)
        
        for move_key in ['up', 'down', 'left', 'right', 'shift']:
            is_active = (move_key == active_move) or (move_key == 'shift' and shift_held)
            if is_active:
                self.key_hold_timers[move_key] += 1
            else:
                self.key_hold_timers[move_key] = 0

            if self.key_hold_timers[move_key] == 1 or \
               (self.key_hold_timers[move_key] > self.DAS_DELAY and self.key_hold_timers[move_key] % self.DAS_RATE == 0):
                if move_key == 'left':
                    self._move_piece(-1, 0)
                elif move_key == 'right':
                    self._move_piece(1, 0)
                elif move_key == 'up' and not shift_held: # Prioritize shift for rotation
                    self._rotate_piece()
                elif move_key == 'shift':
                    self._rotate_piece(clockwise=False)

        # Soft drop
        if self.key_hold_timers['down'] > 0:
            self.drop_counter += self.FPS / 10 # Speed up drop significantly
            self.score += 1 # Small score incentive for soft dropping

        return reward

    def _update_game_state(self):
        self.drop_counter += 1
        if self.drop_counter >= self.drop_speed_frames:
            self.drop_counter = 0
            if not self._move_piece(0, 1):
                # Could not move down, so lock it
                reward = self._lock_shape()
                if not self.game_over:
                    self._spawn_shape()
        
        self._update_particles()

    def _spawn_shape(self, initial=False):
        if initial:
            self.next_shape_id = self.np_random.integers(0, len(self.TETROMINOES))
        
        self.current_shape_id = self.next_shape_id
        self.current_shape = self.TETROMINOES[self.current_shape_id]
        self.next_shape_id = self.np_random.integers(0, len(self.TETROMINOES))
        
        self.current_pos = [self.GRID_WIDTH // 2 - len(self.current_shape[0]) // 2, 0]

        if self._check_collision(self.current_shape, self.current_pos):
            self.game_over = True
            # Place a "ghost" of the piece that failed to spawn
            for r, row in enumerate(self.current_shape):
                for c, cell in enumerate(row):
                    if cell:
                        if 0 <= self.current_pos[1] + r < self.GRID_HEIGHT:
                            self.grid[self.current_pos[1] + r][self.current_pos[0] + c] = self.current_shape_id + 1

    def _move_piece(self, dx, dy):
        new_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        if not self._check_collision(self.current_shape, new_pos):
            self.current_pos = list(new_pos)
            return True
        return False

    def _rotate_piece(self, clockwise=True):
        if self.current_shape_id == 1: return # 'O' shape doesn't rotate
        
        shape = np.array(self.current_shape)
        if clockwise:
            rotated_shape = np.rot90(shape, k=-1).tolist()
        else:
            rotated_shape = np.rot90(shape, k=1).tolist()
        
        # Wall kick tests
        offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        for ox, oy in offsets:
            new_pos = (self.current_pos[0] + ox, self.current_pos[1] + oy)
            if not self._check_collision(rotated_shape, new_pos):
                self.current_shape = rotated_shape
                self.current_pos = list(new_pos)
                # sfx: Rotate
                return True
        return False

    def _check_collision(self, shape, pos):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = pos[0] + c, pos[1] + r
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[y][x] == 0):
                        return True
        return False

    def _lock_shape(self):
        # sfx: Lock piece
        for r, row in enumerate(self.current_shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = self.current_pos[0] + c, self.current_pos[1] + r
                    if 0 <= y < self.GRID_HEIGHT:
                        self.grid[y][x] = self.current_shape_id + 1
        
        return self._clear_lines()

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if all(self.grid[r]):
                lines_to_clear.append(r)

        if not lines_to_clear:
            return 0

        # Create particles
        for r in lines_to_clear:
            # sfx: Line clear
            for i in range(30):
                px = self.GRID_X + self.np_random.integers(0, self.GRID_WIDTH) * self.CELL_SIZE
                py = self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                pvx = (self.np_random.random() - 0.5) * 4
                pvy = (self.np_random.random() - 0.5) * 4
                plife = self.np_random.integers(15, 30)
                pcolor = random.choice([(255,255,255), (255,255,100)])
                self.particles.append([px, py, pvx, pvy, plife, pcolor])

        # Remove lines and shift down
        for r in sorted(lines_to_clear, reverse=True):
            self.grid = np.delete(self.grid, r, axis=0)
        
        new_rows = np.zeros((len(lines_to_clear), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))

        self.lines_cleared += len(lines_to_clear)
        
        # Update drop speed every 2 lines
        self.drop_speed_frames = max(5, self.FPS / (1 + (self.lines_cleared // 2) * 0.2))

        # Calculate reward
        reward_map = {1: 1, 2: 2, 3: 4, 4: 8}
        num_cleared = len(lines_to_clear)
        reward = reward_map.get(num_cleared, 0)
        self.score += (num_cleared ** 2) * 100 # Score based on classic tetris
        
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[4] -= 1    # life -= 1

    def _check_termination(self):
        return (
            self.game_over
            or self.lines_cleared >= self.WIN_CONDITION_LINES
            or self.steps >= self.MAX_STEPS
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared}

    def _render_game(self):
        self._render_grid()
        self._render_locked_pieces()
        if not self.game_over:
            self._render_ghost_piece()
            self._render_current_piece()
        self._render_particles()

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y)
            end_pos = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_block(self, surface, x, y, color_index, is_ghost=False):
        main_color = self.SHAPE_COLORS[color_index]
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.gfxdraw.rectangle(surface, rect, (*main_color, 80))
        else:
            darker_color = tuple(max(0, c - 50) for c in main_color)
            lighter_color = tuple(min(255, c + 50) for c in main_color)
            
            pygame.draw.rect(surface, darker_color, rect)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(surface, main_color, inner_rect)
            
            # Highlight
            pygame.draw.line(surface, lighter_color, (rect.left+2, rect.top+2), (rect.right-3, rect.top+2), 2)
            pygame.draw.line(surface, lighter_color, (rect.left+2, rect.top+2), (rect.left+2, rect.bottom-3), 2)


    def _render_locked_pieces(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] != 0:
                    color_index = self.grid[r][c] - 1
                    self._render_block(self.screen, self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, color_index)

    def _render_current_piece(self):
        for r, row in enumerate(self.current_shape):
            for c, cell in enumerate(row):
                if cell:
                    x = self.GRID_X + (self.current_pos[0] + c) * self.CELL_SIZE
                    y = self.GRID_Y + (self.current_pos[1] + r) * self.CELL_SIZE
                    self._render_block(self.screen, x, y, self.current_shape_id)

    def _render_ghost_piece(self):
        ghost_pos = list(self.current_pos)
        while not self._check_collision(self.current_shape, (ghost_pos[0], ghost_pos[1] + 1)):
            ghost_pos[1] += 1
        
        if ghost_pos[1] > self.current_pos[1]:
            for r, row in enumerate(self.current_shape):
                for c, cell in enumerate(row):
                    if cell:
                        x = self.GRID_X + (ghost_pos[0] + c) * self.CELL_SIZE
                        y = self.GRID_Y + (ghost_pos[1] + r) * self.CELL_SIZE
                        self._render_block(self.screen, x, y, self.current_shape_id, is_ghost=True)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 30.0))))
            color = (*p[5], alpha)
            size = int(8 * (p[4] / 30.0))
            if size > 0:
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size//2, size//2), size//2)
                self.screen.blit(temp_surf, (int(p[0] - size//2), int(p[1] - size//2)))

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score:08d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 50))
        
        # Lines cleared display
        lines_text = self.font_main.render(f"LINES", True, self.COLOR_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.WIDTH - lines_text.get_width() - 20, 20))
        self.screen.blit(lines_val, (self.WIDTH - lines_val.get_width() - 20, 50))

        # Next piece preview
        preview_x, preview_y = self.WIDTH - 120, self.HEIGHT - 120
        preview_box = pygame.Rect(preview_x, preview_y, 100, 100)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, preview_box, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, width=2, border_radius=5)
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (preview_box.centerx - next_text.get_width()//2, preview_box.top + 5))

        next_shape_matrix = self.TETROMINOES[self.next_shape_id]
        shape_w = len(next_shape_matrix[0]) * self.CELL_SIZE
        shape_h = len(next_shape_matrix) * self.CELL_SIZE
        start_x = preview_box.centerx - shape_w // 2
        start_y = preview_box.centery - shape_h // 2 + 10

        for r, row in enumerate(next_shape_matrix):
            for c, cell in enumerate(row):
                if cell:
                    self._render_block(self.screen, start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE, self.next_shape_id)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            text_surf = self.font_main.render(status_text, True, (255, 255, 100))
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()