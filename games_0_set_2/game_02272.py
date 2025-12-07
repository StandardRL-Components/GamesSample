
# Generated: 2025-08-27T19:51:50.175457
# Source Brief: brief_02272.md
# Brief Index: 2272

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop, Shift to hold piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle game. Rotate and drop falling tetrominoes to clear lines and get a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        
        # Positioning the grid in the center
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        # Visuals
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_BG = (30, 30, 45)
        self.COLOR_UI_BORDER = (60, 60, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255)
        
        # Tetromino shapes and colors
        self.TETROMINOES = {
            'I': [[1, 1, 1, 1]],
            'O': [[1, 1], [1, 1]],
            'T': [[0, 1, 0], [1, 1, 1]],
            'J': [[1, 0, 0], [1, 1, 1]],
            'L': [[0, 0, 1], [1, 1, 1]],
            'S': [[0, 1, 1], [1, 1, 0]],
            'Z': [[1, 1, 0], [0, 1, 1]],
        }
        self.TETROMINO_COLORS = {
            1: (50, 200, 200),   # I - Cyan
            2: (220, 220, 50),   # O - Yellow
            3: (180, 50, 180),   # T - Purple
            4: (50, 50, 220),    # J - Blue
            5: (220, 120, 50),   # L - Orange
            6: (50, 220, 50),    # S - Green
            7: (220, 50, 50),    # Z - Red
        }

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 16, bold=True)
        
        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.fall_time = 0
        self.fall_speed = 0
        self.lines_cleared = 0
        self.clear_animation_timer = 0
        self.lines_to_clear = []
        self.prev_up_pressed = False
        self.prev_space_pressed = False
        self.prev_shift_pressed = False
        self.move_cooldown = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Grid: 10 wide, 20 high, with 4 buffer rows at the top for spawning
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT + 4), dtype=int)
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.fall_time = 0
        self.fall_speed = 1.0  # Time in seconds for one cell drop

        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()
        self.held_piece = None
        self.can_hold = True

        self.clear_animation_timer = 0
        self.lines_to_clear = []
        
        # For handling one-shot actions
        self.prev_up_pressed = False
        self.prev_space_pressed = False
        self.prev_shift_pressed = False
        self.move_cooldown = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage faster play
        self.steps += 1
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Handle line clear animation delay
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._finalize_line_clear()
        else:
            # --- Handle Player Input ---
            input_reward = self._handle_input(movement, space_held, shift_held)
            reward += input_reward
            
            # --- Handle Gravity ---
            self.fall_time += 1 / 30.0  # Assuming 30 FPS step rate
            if self.fall_time >= self.fall_speed:
                self.fall_time = 0
                self.current_piece['y'] += 1
                if not self._is_valid_position():
                    self.current_piece['y'] -= 1
                    lock_reward = self._lock_piece()
                    reward += lock_reward
                    # sfx_lock

        # --- Check for Termination ---
        terminated = self.game_over or self.score >= 500 or self.steps >= 10000
        if terminated and self.score >= 500 and not self.game_over:
            reward += 100 # Victory bonus
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        # --- One-shot actions (triggered on rising edge) ---
        up_pressed = (movement == 1)
        if up_pressed and not self.prev_up_pressed:
            self._rotate_piece()
            # sfx_rotate
        self.prev_up_pressed = up_pressed

        if space_held and not self.prev_space_pressed:
            reward += self._hard_drop()
            # sfx_hard_drop
        self.prev_space_pressed = space_held

        if shift_held and not self.prev_shift_pressed:
            if self._hold_piece():
                # sfx_hold
                pass
        self.prev_shift_pressed = shift_held

        # --- Continuous actions (movement) ---
        self.move_cooldown = max(0, self.move_cooldown - 1)
        if self.move_cooldown == 0:
            moved = False
            if movement == 3: # Left
                self.current_piece['x'] -= 1
                if not self._is_valid_position(): self.current_piece['x'] += 1
                else: moved = True
            elif movement == 4: # Right
                self.current_piece['x'] += 1
                if not self._is_valid_position(): self.current_piece['x'] -= 1
                else: moved = True
            
            if moved:
                # sfx_move
                self.move_cooldown = 3 # 3-frame cooldown between moves
        
        # Soft drop
        if movement == 2:
            self.fall_speed = 0.05 # Faster fall speed while held
        else:
            self.fall_speed = max(0.1, 1.0 - (self.lines_cleared // 50) * 0.05)
        
        return reward

    def _get_piece_shape(self, piece):
        base_shape = self.TETROMINOES[piece['shape']]
        # Rotate shape
        rotated_shape = base_shape
        for _ in range(piece['rotation'] % len(base_shape)):
             rotated_shape = list(zip(*rotated_shape[::-1]))
        # This rotation logic is for square matrices. The shapes are not square.
        # A better way is to pre-calculate rotations. For now, let's use a simpler method.
        shape = base_shape
        for _ in range(piece['rotation']):
            shape = [list(row) for row in zip(*shape[::-1])]
        return shape

    def _is_valid_position(self, piece=None, offset_x=0, offset_y=0):
        if piece is None: piece = self.current_piece
        shape = self._get_piece_shape(piece)
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x = piece['x'] + c + offset_x
                    y = piece['y'] + r + offset_y
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT + 4 and self.grid[x, y] == 0):
                        return False
        return True

    def _rotate_piece(self):
        self.current_piece['rotation'] = (self.current_piece['rotation'] + 1) % 4
        if not self._is_valid_position():
            # Wall kick: try moving left/right
            self.current_piece['x'] -= 1
            if self._is_valid_position(): return
            self.current_piece['x'] += 2
            if self._is_valid_position(): return
            self.current_piece['x'] -= 1
            # If all fail, revert rotation
            self.current_piece['rotation'] = (self.current_piece['rotation'] - 1 + 4) % 4

    def _hard_drop(self):
        original_y = self.current_piece['y']
        while self._is_valid_position(offset_y=1):
            self.current_piece['y'] += 1
        
        # Small reward for distance dropped
        drop_distance = self.current_piece['y'] - original_y
        reward = drop_distance * 0.02
        
        self.fall_time = self.fall_speed # Trigger lock immediately
        return reward

    def _hold_piece(self):
        if not self.can_hold: return False
        self.can_hold = False
        
        if self.held_piece is None:
            self.held_piece = self.current_piece
            self.current_piece = self._new_piece()
        else:
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self._get_piece_shape(self.current_piece)[0]) // 2
            self.current_piece['y'] = 2
            self.current_piece['rotation'] = 0
        
        if not self._is_valid_position(): # if swap results in collision, it's game over
            self.game_over = True
        return True

    def _lock_piece(self):
        shape = self._get_piece_shape(self.current_piece)
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = self.current_piece['x'] + c, self.current_piece['y'] + r
                    if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT + 4:
                        self.grid[x, y] = self.current_piece['type_id']
        
        hole_penalty = self._calculate_hole_penalty()
        clear_reward = self._check_line_clears()
        
        # Spawn new piece
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()
        self.can_hold = True
        
        if not self._is_valid_position():
            self.game_over = True
            
        return hole_penalty + clear_reward

    def _calculate_hole_penalty(self):
        penalty = 0
        shape = self._get_piece_shape(self.current_piece)
        for c_offset, col in enumerate(np.transpose(shape)):
            if np.any(col):
                px = self.current_piece['x'] + c_offset
                py_bottom = self.current_piece['y'] + np.where(col)[0][-1]
                for y in range(py_bottom + 1, self.GRID_HEIGHT + 4):
                    if 0 <= px < self.GRID_WIDTH:
                        if self.grid[px, y] == 0: penalty -= 1
                        else: break
        return penalty

    def _check_line_clears(self):
        full_lines = []
        for r in range(self.GRID_HEIGHT + 4):
            is_full = True
            for c in range(self.GRID_WIDTH):
                if self.grid[c, r] == 0:
                    is_full = False
                    break
            if is_full:
                full_lines.append(r)

        if full_lines:
            self.lines_to_clear = full_lines
            self.clear_animation_timer = 6 # frames for animation
            # sfx_line_clear
            
            line_scores = [0, 100, 300, 500, 800] # 0, 1, 2, 3, 4 lines
            self.score += line_scores[min(len(full_lines), 4)]
            
            reward_map = {1: 10, 2: 25, 3: 40, 4: 50}
            return reward_map.get(len(full_lines), 50)
        return 0

    def _finalize_line_clear(self):
        if not self.lines_to_clear: return
        self.lines_to_clear.sort(reverse=True)
        
        for r in self.lines_to_clear:
            self.grid = np.delete(self.grid, r, axis=1)
            new_row = np.zeros((self.GRID_WIDTH, 1), dtype=int)
            self.grid = np.insert(self.grid, 0, new_row, axis=1)
        
        self.lines_cleared += len(self.lines_to_clear)
        self.lines_to_clear = []

    def _new_piece(self):
        shape_name = self.np_random.choice(list(self.TETROMINOES.keys()))
        type_id = list(self.TETROMINOES.keys()).index(shape_name) + 1
        piece = {
            'shape': shape_name,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - len(self.TETROMINOES[shape_name][0]) // 2,
            'y': 2, # Spawn in buffer zone
            'type_id': type_id
        }
        return piece

    def _get_ghost_piece_y(self):
        if not self.current_piece: return 0
        y = self.current_piece['y']
        while self._is_valid_position(offset_y=y - self.current_piece['y'] + 1):
            y += 1
        return y

    def _draw_block(self, surface, x, y, color, alpha=255):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        light_color = tuple(min(255, c + 60) for c in color)
        dark_color = tuple(max(0, c - 60) for c in color)

        if alpha < 255:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*color, alpha))
            surface.blit(s, (x, y))
            pygame.draw.rect(surface, (*light_color, alpha), rect, 1)
        else:
            pygame.draw.rect(surface, dark_color, rect)
            pygame.draw.rect(surface, color, rect.inflate(-3, -3))
            pygame.draw.line(surface, light_color, rect.topleft, (rect.right - 1, rect.top))
            pygame.draw.line(surface, light_color, rect.topleft, (rect.left, rect.bottom - 1))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, grid_rect)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET), (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + i * self.CELL_SIZE))

        # Draw locked pieces
        for x in range(self.GRID_WIDTH):
            for y in range(4, self.GRID_HEIGHT + 4): # Only draw visible portion
                if self.grid[x, y] != 0:
                    color = self.TETROMINO_COLORS[self.grid[x, y]]
                    self._draw_block(self.screen, self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + (y - 4) * self.CELL_SIZE, color)
        
        # Draw ghost piece
        ghost_y = self._get_ghost_piece_y()
        shape = self._get_piece_shape(self.current_piece)
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell and ghost_y + r >= 4:
                    self._draw_block(self.screen, self.GRID_X_OFFSET + (self.current_piece['x'] + c) * self.CELL_SIZE, self.GRID_Y_OFFSET + (ghost_y + r - 4) * self.CELL_SIZE, self.COLOR_GHOST, 60)

        # Draw current piece
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell and self.current_piece['y'] + r >= 4:
                    color = self.TETROMINO_COLORS[self.current_piece['type_id']]
                    self._draw_block(self.screen, self.GRID_X_OFFSET + (self.current_piece['x'] + c) * self.CELL_SIZE, self.GRID_Y_OFFSET + (self.current_piece['y'] + r - 4) * self.CELL_SIZE, color)

        # Draw line clear animation
        if self.clear_animation_timer > 0:
            flash_color = (255, 255, 255, 150 + 100 * math.sin(self.clear_animation_timer * math.pi / 6))
            for r in self.lines_to_clear:
                if r >= 4:
                    rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET + (r - 4) * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                    pygame.gfxdraw.box(self.screen, rect, flash_color)

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GRID_X_OFFSET - 120, self.GRID_Y_OFFSET + 20))
        self.screen.blit(score_val, (self.GRID_X_OFFSET - 120, self.GRID_Y_OFFSET + 45))

        # --- Next Piece Display ---
        self._render_side_panel(self.next_piece, "NEXT", self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y_OFFSET)
        
        # --- Held Piece Display ---
        self._render_side_panel(self.held_piece, "HOLD", self.GRID_X_OFFSET - 130, self.GRID_Y_OFFSET + 120)

    def _render_side_panel(self, piece, title, x, y):
        panel_rect = pygame.Rect(x, y, 110, 90)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, panel_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_BORDER, panel_rect, 2)
        
        title_surf = self.font_title.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (panel_rect.centerx - title_surf.get_width() // 2, y + 8))

        if piece:
            shape = self._get_piece_shape(piece)
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            start_x = panel_rect.centerx - shape_w // 2
            start_y = panel_rect.centery - shape_h // 2 + 10

            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        color = self.TETROMINO_COLORS[piece['type_id']]
                        self._draw_block(self.screen, start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared}

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