
# Generated: 2025-08-28T01:27:47.197332
# Source Brief: brief_04110.md
# Brief Index: 4110

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. "
        "Space to hard drop, Shift to hold a piece."
    )

    game_description = (
        "A fast-paced falling block puzzle. "
        "Clear 10 lines to win, but don't let the stack reach the top!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.SIDE_PANEL_WIDTH = 150
        
        self.GRID_PIXEL_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GRID_PIXEL_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_PIXEL_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_PIXEL_HEIGHT) // 2
        
        self.MAX_STEPS = 10000
        self.WIN_CONDITION_LINES = 10
        
        self.INITIAL_FALL_TIME = 30  # In frames (30 frames = 1 second)
        self.FALL_TIME_REDUCTION_PER_LINE = 1.5
        self.SOFT_DROP_MULTIPLIER = 5.0

        # --- Colors ---
        self.COLOR_BG = (25, 25, 40)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PANEL = (20, 20, 35)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255)
        self.PIECE_COLORS = [
            (230, 60, 60),    # Z
            (60, 230, 60),    # S
            (60, 60, 230),    # J
            (230, 150, 60),   # L
            (230, 230, 60),   # O
            (150, 60, 230),   # T
            (60, 230, 230),   # I
        ]

        # --- Tetromino Shapes ---
        self.PIECES = [
            [[1, 1, 0], [0, 1, 1]],  # Z
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 1], [1, 1]],        # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[1, 1, 1, 1]]           # I
        ]
        
        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece_idx = None
        self.held_piece_idx = None
        self.can_hold = None
        self.piece_bag = None
        self.fall_timer = None
        self.fall_speed = None
        self.lines_cleared = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.reward_this_step = 0
        self.lines_to_clear = None
        self.line_clear_timer = 0
        
        # Input handling state
        self.prev_action = np.array([0, 0, 0])
        self.move_timer = 0
        self.MOVE_DELAY_INITIAL = 10 # frames
        self.MOVE_DELAY_REPEAT = 3 # frames

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.piece_bag = list(range(len(self.PIECES)))
        random.shuffle(self.piece_bag)
        
        self.held_piece_idx = -1
        self.can_hold = True
        
        self._spawn_piece()
        self._spawn_piece() # First is current, second is next

        self.fall_timer = 0
        self.fall_speed = self.INITIAL_FALL_TIME
        self.reward_this_step = 0
        self.lines_to_clear = []
        self.line_clear_timer = 0

        self.prev_action = np.array([0, 0, 0])
        self.move_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.reward_this_step = -0.01  # Small penalty for time passing

        if not self.game_over:
            if self.line_clear_timer > 0:
                self.line_clear_timer -= 1
                if self.line_clear_timer == 0:
                    self._execute_line_clear()
            else:
                self._handle_input(action)
                self._update_gravity(action)

        terminated = self._check_termination()
        
        if terminated and not self.game_over: # Win condition
            self.reward_this_step += 100
            self.game_over = True
        elif self.game_over and self.lines_cleared < self.WIN_CONDITION_LINES: # Loss
             self.reward_this_step -= 100

        self.prev_action = action

        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_btn, shift_btn = action
        prev_movement, prev_space_btn, prev_shift_btn = self.prev_action

        # --- Discrete actions (on press) ---
        if movement == 1 and prev_movement != 1: self._rotate_piece() # Rotate
        if space_btn == 1 and prev_space_btn == 0: self._hard_drop() # Hard Drop
        if shift_btn == 1 and prev_shift_btn == 0: self._hold_piece() # Hold

        # --- Continuous actions (while held) ---
        is_moving_horizontally = movement in [3, 4]
        prev_moving_horizontally = prev_movement in [3, 4]

        if is_moving_horizontally:
            if not prev_moving_horizontally or movement != prev_movement:
                self.move_timer = self.MOVE_DELAY_INITIAL
                dx = -1 if movement == 3 else 1
                self._move(dx, 0)
            else:
                self.move_timer -= 1
                if self.move_timer <= 0:
                    self.move_timer = self.MOVE_DELAY_REPEAT
                    dx = -1 if movement == 3 else 1
                    self._move(dx, 0)
        else:
            self.move_timer = 0

    def _update_gravity(self, action):
        if self.current_piece is None:
            return

        movement = action[0]
        fall_increment = self.SOFT_DROP_MULTIPLIER if movement == 2 else 1
        self.fall_timer += fall_increment

        if self.fall_timer >= self.fall_speed:
            self.fall_timer = 0
            if not self._move(0, 1):
                self._lock_piece()

    def _spawn_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.PIECES)))
            random.shuffle(self.piece_bag)

        piece_idx = self.piece_bag.pop(0)
        
        if self.current_piece is None: # First piece
            self.current_piece = {
                "idx": piece_idx,
                "shape": self.PIECES[piece_idx],
                "x": self.GRID_WIDTH // 2 - len(self.PIECES[piece_idx][0]) // 2,
                "y": 0
            }
        else: # Subsequent pieces
            self.current_piece = self.next_piece_idx
            self.current_piece["x"] = self.GRID_WIDTH // 2 - len(self.current_piece["shape"][0]) // 2
            self.current_piece["y"] = 0
            
        self.next_piece_idx = {
            "idx": piece_idx,
            "shape": self.PIECES[piece_idx]
        }
        
        # Check for game over
        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            self.current_piece = None

    def _lock_piece(self):
        if self.current_piece is None: return
        
        # Calculate placement reward
        supporters = 0
        piece_y, piece_x = self.current_piece['y'], self.current_piece['x']
        shape = self.current_piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    # Check cell directly below
                    if piece_y + r + 1 >= self.GRID_HEIGHT or self.grid[piece_y + r + 1, piece_x + c] > 0:
                        supporters +=1
        
        if supporters == 1: self.reward_this_step += 2 # Risky
        elif supporters > 1: self.reward_this_step -= 0.2 # Safe

        # Place piece on grid
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[piece_y + r, piece_x + c] = self.current_piece["idx"] + 1

        # SFX: pygame.mixer.Sound('lock.wav').play()
        self.current_piece = None
        self.can_hold = True
        self._check_for_line_clears()
        
        if not self.lines_to_clear:
            self._spawn_piece()

    def _check_for_line_clears(self):
        full_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] > 0):
                full_rows.append(r)
        
        if full_rows:
            self.lines_to_clear = full_rows
            self.line_clear_timer = 10 # frames for animation
            self.reward_this_step += len(full_rows) * 1 # +1 per line
            self.score += len(full_rows) * 100
            # SFX: pygame.mixer.Sound('clear.wav').play()
        
    def _execute_line_clear(self):
        num_cleared = len(self.lines_to_clear)
        for r in sorted(self.lines_to_clear, reverse=True):
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        
        self.lines_cleared += num_cleared
        self.fall_speed = max(5, self.INITIAL_FALL_TIME - self.lines_cleared * self.FALL_TIME_REDUCTION_PER_LINE)
        self.lines_to_clear = []
        self._spawn_piece()

    def _move(self, dx, dy):
        if self.current_piece is None: return False
        
        test_pos = self.current_piece.copy()
        test_pos["x"] += dx
        test_pos["y"] += dy
        
        if self._is_valid_position(test_pos):
            self.current_piece = test_pos
            return True
        return False

    def _rotate_piece(self):
        if self.current_piece is None: return
        
        original_shape = self.current_piece["shape"]
        rotated_shape = list(zip(*original_shape[::-1]))

        test_piece = self.current_piece.copy()
        test_piece["shape"] = rotated_shape
        
        # Wall kick checks
        for dx in [0, 1, -1, 2, -2]:
            if self._is_valid_position(test_piece, offset=(dx, 0)):
                self.current_piece["shape"] = rotated_shape
                self.current_piece["x"] += dx
                # SFX: pygame.mixer.Sound('rotate.wav').play()
                return

    def _hard_drop(self):
        if self.current_piece is None: return

        while self._move(0, 1):
            self.score += 2 # Small bonus for hard dropping
        self._lock_piece()
        # SFX: pygame.mixer.Sound('hard_drop.wav').play()

    def _hold_piece(self):
        if not self.can_hold or self.current_piece is None: return
        
        # SFX: pygame.mixer.Sound('hold.wav').play()
        if self.held_piece_idx == -1:
            self.held_piece_idx = self.current_piece["idx"]
            self._spawn_piece()
        else:
            current_idx = self.current_piece["idx"]
            held_idx = self.held_piece_idx
            
            self.held_piece_idx = current_idx
            self.current_piece = {
                "idx": held_idx,
                "shape": self.PIECES[held_idx],
                "x": self.GRID_WIDTH // 2 - len(self.PIECES[held_idx][0]) // 2,
                "y": 0
            }
            if not self._is_valid_position(self.current_piece):
                self.game_over = True
                self.current_piece = None
        
        self.can_hold = False

    def _is_valid_position(self, piece, offset=(0, 0)):
        shape = piece["shape"]
        pos_x, pos_y = piece["x"] + offset[0], piece["y"] + offset[1]
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_y, grid_x = pos_y + r, pos_x + c
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_y, grid_x] > 0:
                        return False
        return True

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS or self.lines_cleared >= self.WIN_CONDITION_LINES

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT))
        
        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    color_idx = int(self.grid[r, c] - 1)
                    self._draw_cell(c, r, self.PIECE_COLORS[color_idx])

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece):
                ghost_piece["y"] += 1
            ghost_piece["y"] -= 1
            self._draw_piece(ghost_piece, ghost=True)
            
        # Draw current piece
        if self.current_piece and not self.game_over:
            self._draw_piece(self.current_piece)
            
        # Draw line clear animation
        if self.line_clear_timer > 0:
            flash_alpha = 150 * (self.line_clear_timer / 10)
            flash_color = (255, 255, 255, flash_alpha)
            for r in self.lines_to_clear:
                s = pygame.Surface((self.GRID_PIXEL_WIDTH, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(flash_color)
                self.screen.blit(s, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE))

        # Draw grid border
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_PIXEL_WIDTH, self.GRID_PIXEL_HEIGHT), 2)

    def _render_ui(self):
        # Left Panel
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (0, 0, self.SIDE_PANEL_WIDTH, self.HEIGHT))
        score_text = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 30))
        self.screen.blit(score_val, (20, 60))
        
        lines_text = self.font_main.render("LINES", True, self.COLOR_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 120))
        self.screen.blit(lines_val, (20, 150))
        
        # Right Panel
        pygame.draw.rect(self.screen, self.COLOR_PANEL, (self.WIDTH - self.SIDE_PANEL_WIDTH, 0, self.SIDE_PANEL_WIDTH, self.HEIGHT))
        
        # Next Piece
        next_text = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - self.SIDE_PANEL_WIDTH + 20, 30))
        if self.next_piece_idx:
            self._draw_piece_in_ui(self.next_piece_idx, self.WIDTH - self.SIDE_PANEL_WIDTH + 25, 60)
            
        # Held Piece
        hold_text = self.font_main.render("HOLD", True, self.COLOR_TEXT)
        self.screen.blit(hold_text, (self.WIDTH - self.SIDE_PANEL_WIDTH + 20, 200))
        if self.held_piece_idx != -1:
            held_piece_data = {"idx": self.held_piece_idx, "shape": self.PIECES[self.held_piece_idx]}
            self._draw_piece_in_ui(held_piece_data, self.WIDTH - self.SIDE_PANEL_WIDTH + 25, 230)

        # Game Over / Win Text
        if self.game_over:
            msg = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            text_surf = self.font_main.render(msg, True, (255, 80, 80))
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20,20))
            self.screen.blit(text_surf, text_rect)

    def _draw_piece(self, piece, ghost=False):
        shape = piece["shape"]
        color = self.PIECE_COLORS[piece["idx"]]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    if ghost:
                        rect = (
                            self.GRID_X + (piece["x"] + c) * self.CELL_SIZE,
                            self.GRID_Y + (piece["y"] + r) * self.CELL_SIZE,
                            self.CELL_SIZE,
                            self.CELL_SIZE
                        )
                        pygame.draw.rect(self.screen, self.COLOR_GHOST, rect, 2)
                    else:
                        self._draw_cell(piece["x"] + c, piece["y"] + r, color)

    def _draw_piece_in_ui(self, piece_data, start_x, start_y):
        shape = piece_data["shape"]
        color = self.PIECE_COLORS[piece_data["idx"]]
        ui_cell_size = self.CELL_SIZE * 0.9
        
        # Center the piece in a 4x4 box
        shape_w = len(shape[0]) * ui_cell_size
        start_x += (4 * ui_cell_size - shape_w) / 2

        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(c, r, color, offset=(start_x, start_y), cell_size=ui_cell_size, use_grid=False)

    def _draw_cell(self, grid_c, grid_r, color, offset=(0,0), cell_size=None, use_grid=True):
        if cell_size is None: cell_size = self.CELL_SIZE
        
        if use_grid:
            x = self.GRID_X + grid_c * cell_size
            y = self.GRID_Y + grid_r * cell_size
        else:
            x = offset[0] + grid_c * cell_size
            y = offset[1] + grid_r * cell_size

        main_rect = (x + 1, y + 1, cell_size - 2, cell_size - 2)
        
        # Create a beveled 3D look
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        
        pygame.draw.rect(self.screen, dark_color, (x, y, cell_size, cell_size))
        pygame.draw.polygon(self.screen, light_color, [(x, y), (x + cell_size, y), (x + cell_size -1, y+1), (x+1, y+cell_size-1), (x, y+cell_size)])
        pygame.draw.rect(self.screen, color, main_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")