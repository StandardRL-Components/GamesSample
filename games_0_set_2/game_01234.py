
# Generated: 2025-08-27T16:29:02.157913
# Source Brief: brief_01234.md
# Brief Index: 1234

        
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
        "Controls: ←→ to move, ↑/Shift to rotate, ↓ for soft drop, Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Place falling blocks to clear lines and score points before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        
        self.BOARD_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.BOARD_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        
        self.BOARD_X_OFFSET = (self.SCREEN_WIDTH - self.BOARD_WIDTH) // 2
        self.BOARD_Y_OFFSET = (self.SCREEN_HEIGHT - self.BOARD_HEIGHT) // 2

        self.MAX_STEPS = 10000
        self.TARGET_LINES = 10
        self.FPS = 30 # For auto_advance=True, this is the assumed step rate

        # --- Colors ---
        self.COLOR_BG = (26, 26, 46) # Dark blue
        self.COLOR_GRID = (42, 42, 78) # Lighter blue
        self.COLOR_TEXT = (230, 230, 255)
        self.COLOR_GHOST = (255, 255, 255, 60)
        self.COLOR_FLASH = (255, 255, 255)
        self.PIECE_COLORS = [
            (239, 131, 84),  # L-piece (Orange)
            (66, 179, 229),  # J-piece (Blue)
            (149, 216, 91),  # S-piece (Green)
            (250, 204, 92),  # Z-piece (Yellow)
            (172, 126, 223), # T-piece (Purple)
            (234, 91, 137),  # I-piece (Pink)
            (93, 225, 208),  # O-piece (Cyan)
        ]

        # --- Piece Shapes (Tetrominos) ---
        self.PIECE_SHAPES = {
            0: [[(0, 0), (-1, 0), (1, 0), (1, -1)]],  # L
            1: [[(0, 0), (-1, 0), (1, 0), (-1, -1)]], # J
            2: [[(0, 0), (-1, 0), (0, -1), (1, -1)]], # S
            3: [[(0, 0), (1, 0), (0, -1), (-1, -1)]], # Z
            4: [[(0, 0), (-1, 0), (1, 0), (0, -1)]],  # T
            5: [[(0, 0), (-1, 0), (1, 0), (2, 0)], [(0, 0), (0, -1), (0, 1), (0, 2)]], # I
            6: [[(0, 0), (1, 0), (0, -1), (1, -1)]], # O
        }
        
        # --- Gymnasium Setup ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.board = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0.0
        self.fall_timer = 0
        self.last_action = np.zeros(3, dtype=int)
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        self.np_random = None

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_speed = 1.0 # cells per second
        self.fall_timer = 0
        
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        
        self.last_action = np.zeros(3, dtype=int)
        
        self.current_piece = self._spawn_piece()
        self.next_piece = self._spawn_piece()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Handle Line Clear Animation ---
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._execute_line_clear()
            # No other actions or physics while clearing
            terminated = self._check_termination()
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Unpack Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input (with debouncing for rotations) ---
        # Movement
        if movement == 3: # Left
            self._move(-1)
        elif movement == 4: # Right
            self._move(1)
        
        # Rotation
        if movement == 1 and self.last_action[0] != 1: # Rotate CW on 'Up' press
            self._rotate(1)
        if shift_held and not (self.last_action[2] == 1): # Rotate CCW on 'Shift' press
            self._rotate(-1)
        
        self.last_action = action

        # --- Handle Hard Drop ---
        if space_held:
            # Sound: Hard drop
            holes, height = self._hard_drop()
            reward += 0.1 * height - 0.01 * holes # Placement reward
            lines_cleared_count = self._check_and_start_line_clear()
            if lines_cleared_count > 0:
                reward += [1, 2, 4, 8][lines_cleared_count - 1] # Line clear reward
            self._spawn_new_round()
        else:
            # --- Handle Soft Drop & Gravity ---
            soft_drop_multiplier = 4.0 if movement == 2 else 1.0
            self.fall_timer += self.fall_speed * soft_drop_multiplier
            
            if self.fall_timer >= self.FPS:
                self.fall_timer -= self.FPS
                self.current_piece['y'] += 1
                # Sound: Tick
                
                if not self._is_valid_position():
                    self.current_piece['y'] -= 1
                    # Sound: Block placed
                    holes, height = self._lock_piece()
                    reward += 0.1 * height - 0.01 * holes # Placement reward
                    lines_cleared_count = self._check_and_start_line_clear()
                    if lines_cleared_count > 0:
                        reward += [1, 2, 4, 8][lines_cleared_count - 1] # Line clear reward
                    self._spawn_new_round()

        terminated = self._check_termination()
        if terminated:
            if self.lines_cleared >= self.TARGET_LINES:
                reward += 100 # Win bonus
            elif self.game_over:
                reward -= 100 # Loss penalty

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}

    # --- Game Logic Helpers ---

    def _spawn_new_round(self):
        """Handles state changes after a piece is locked."""
        self.current_piece = self.next_piece
        self.next_piece = self._spawn_piece()
        
        if not self._is_valid_position():
            self.game_over = True
            # Sound: Game over

    def _spawn_piece(self):
        shape_idx = self.np_random.integers(0, len(self.PIECE_SHAPES))
        return {
            'shape_idx': shape_idx,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2,
            'y': 0,
            'color_idx': shape_idx,
        }

    def _get_piece_coords(self, piece):
        shape_template = self.PIECE_SHAPES[piece['shape_idx']]
        shape = shape_template[piece['rotation'] % len(shape_template)]
        return [(p[0] + piece['x'], p[1] + piece['y']) for p in shape]

    def _is_valid_position(self, piece=None, offset=(0, 0)):
        if piece is None:
            piece = self.current_piece
        
        coords = self._get_piece_coords(piece)
        for x, y in coords:
            x_offset, y_offset = offset
            x += x_offset
            y += y_offset
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if self.board[y, x] != 0:
                return False
        return True

    def _move(self, dx):
        self.current_piece['x'] += dx
        if not self._is_valid_position():
            self.current_piece['x'] -= dx
        # else: Sound: Move

    def _rotate(self, direction):
        original_rotation = self.current_piece['rotation']
        self.current_piece['rotation'] = (self.current_piece['rotation'] + direction)
        
        if not self._is_valid_position():
            # Basic wall kick: try moving left/right
            original_x = self.current_piece['x']
            for dx in [1, -1, 2, -2]:
                self.current_piece['x'] = original_x + dx
                if self._is_valid_position():
                    # Sound: Rotate
                    return
            # If all kicks fail, revert
            self.current_piece['x'] = original_x
            self.current_piece['rotation'] = original_rotation
        # else: Sound: Rotate

    def _lock_piece(self):
        coords = self._get_piece_coords(self.current_piece)
        min_y, max_y = self.GRID_HEIGHT, -1
        
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.board[y, x] = self.current_piece['color_idx'] + 1
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        # Calculate holes for reward
        holes = 0
        unique_cols = set(int(x) for x, y in coords if 0 <= x < self.GRID_WIDTH)
        for col in unique_cols:
            col_min_y = self.GRID_HEIGHT
            for x, y in coords:
                if int(x) == col:
                    col_min_y = min(col_min_y, y)
            
            for r in range(int(col_min_y) + 1, self.GRID_HEIGHT):
                if self.board[r, col] == 0:
                    holes += 1
        
        height_reward = self.GRID_HEIGHT - min_y if min_y != self.GRID_HEIGHT else 0
        return holes, height_reward

    def _hard_drop(self):
        while self._is_valid_position():
            self.current_piece['y'] += 1
        self.current_piece['y'] -= 1
        return self._lock_piece()

    def _check_and_start_line_clear(self):
        self.lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.board[r, :] != 0):
                self.lines_to_clear.append(r)
        
        if self.lines_to_clear:
            self.clear_animation_timer = 10 # frames
            # Sound: Line clear start
        
        return len(self.lines_to_clear)

    def _execute_line_clear(self):
        num_cleared = len(self.lines_to_clear)
        if num_cleared == 0: return

        # Score based on number of lines cleared at once
        score_bonuses = {1: 40, 2: 100, 3: 300, 4: 1200}
        self.score += score_bonuses.get(num_cleared, 0)
        
        # Remove lines from top to bottom to avoid index issues
        for row_idx in sorted(self.lines_to_clear, reverse=False):
            self.board = np.delete(self.board, row_idx, axis=0)
            new_row = np.zeros((1, self.GRID_WIDTH), dtype=int)
            self.board = np.vstack([new_row, self.board])

        prev_lines = self.lines_cleared
        self.lines_cleared += num_cleared
        
        # Update fall speed every 2 lines
        if self.lines_cleared // 2 > prev_lines // 2:
            self.fall_speed += 0.05
        
        self.lines_to_clear = []

    def _check_termination(self):
        return (self.game_over or 
                self.steps >= self.MAX_STEPS or 
                self.lines_cleared >= self.TARGET_LINES)

    # --- Rendering Helpers ---

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = self.BOARD_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.BOARD_Y_OFFSET), (px, self.BOARD_Y_OFFSET + self.BOARD_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.BOARD_Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, py), (self.BOARD_X_OFFSET + self.BOARD_WIDTH, py))
            
        # Draw locked blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.board[y, x] != 0:
                    self._draw_cell(x, y, self.PIECE_COLORS[int(self.board[y, x] - 1)])
        
        # Draw flashing lines
        if self.clear_animation_timer > 0:
            flash_color = self.COLOR_FLASH if (self.clear_animation_timer // 2) % 2 == 0 else self.COLOR_GRID
            for y in self.lines_to_clear:
                for x in range(self.GRID_WIDTH):
                    self._draw_cell(x, y, flash_color)

        if self.game_over:
            return

        # Draw ghost piece
        ghost_piece = self.current_piece.copy()
        while self._is_valid_position(ghost_piece):
            ghost_piece['y'] += 1
        ghost_piece['y'] -= 1
        for x, y in self._get_piece_coords(ghost_piece):
            self._draw_cell(x, y, self.COLOR_GHOST, is_ghost=True)
            
        # Draw current piece
        piece_color = self.PIECE_COLORS[self.current_piece['color_idx']]
        for x, y in self._get_piece_coords(self.current_piece):
            self._draw_cell(x, y, piece_color)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        px = self.BOARD_X_OFFSET + grid_x * self.CELL_SIZE
        py = self.BOARD_Y_OFFSET + grid_y * self.CELL_SIZE
        
        if not (0 <= grid_y < self.GRID_HEIGHT): return
        
        rect = pygame.Rect(int(px), int(py), self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            # Draw a transparent rectangle for the ghost
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (rect.x, rect.y))
            pygame.gfxdraw.rectangle(self.screen, rect, (255, 255, 255, 100))
        else:
            # Draw a solid block with a border
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 50))
        
        # Lines
        lines_text = self.font_large.render(f"LINES", True, self.COLOR_TEXT)
        lines_val = self.font_large.render(f"{self.lines_cleared} / {self.TARGET_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - 120, 20))
        self.screen.blit(lines_val, (self.SCREEN_WIDTH - 120, 50))

        # Next Piece
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.BOARD_X_OFFSET + self.BOARD_WIDTH + 20, self.BOARD_Y_OFFSET))
        if self.next_piece:
            next_color = self.PIECE_COLORS[self.next_piece['color_idx']]
            shape_template = self.PIECE_SHAPES[self.next_piece['shape_idx']]
            shape = shape_template[0] # Use default rotation
            for p in shape:
                px = self.BOARD_X_OFFSET + self.BOARD_WIDTH + 45 + p[0] * self.CELL_SIZE
                py = self.BOARD_Y_OFFSET + 40 + p[1] * self.CELL_SIZE
                rect = pygame.Rect(int(px), int(py), self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, next_color, rect)
                pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.lines_cleared >= self.TARGET_LINES else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_FLASH)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Block Breaker")
    
    action = env.action_space.sample()
    action.fill(0)

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        
        # Movement
        mov = 0
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        # Buttons
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = np.array([mov, space, shift])

        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines']}")
            # In a real scenario, you might wait for a key press to reset
            # For this demo, we'll just let it sit on the game over screen
            # obs, info = env.reset() # uncomment to auto-reset

        # --- Rendering ---
        # The observation is already the rendered image
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    pygame.quit()