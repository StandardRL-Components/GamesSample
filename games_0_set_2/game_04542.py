
# Generated: 2025-08-28T02:42:44.008135
# Source Brief: brief_04542.md
# Brief Index: 4542

        
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
        "Controls: ←→ to move, ↓ for soft drop, ↑ for clockwise rotation. "
        "Hold Shift for counter-clockwise rotation. Press Space for hard drop."
    )

    game_description = (
        "A fast-paced puzzle game. Rotate and drop falling blocks to clear lines. "
        "Clear 100 lines to win. The game ends if the blocks stack to the top."
    )

    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_X_OFFSET, GRID_Y_OFFSET = 210, 0
    
    CELL_SIZE = 20
    
    # Colors (Neon on Dark)
    COLOR_BG = (10, 10, 25)
    COLOR_GRID = (30, 30, 60)
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_UI_HEADER = (255, 255, 255)
    COLOR_WHITE = (255, 255, 255)
    
    PIECE_COLORS = [
        (0, 255, 255),  # I (Cyan)
        (255, 255, 0),  # O (Yellow)
        (170, 0, 255),  # T (Purple)
        (0, 0, 255),    # J (Blue)
        (255, 128, 0),  # L (Orange)
        (0, 255, 0),    # S (Green)
        (255, 0, 0),    # Z (Red)
    ]

    # Piece shapes (tetrominoes)
    PIECES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1], [1, 1]],  # O
        [[0, 1, 0], [1, 1, 1]],  # T
        [[1, 0, 0], [1, 1, 1]],  # J
        [[0, 0, 1], [1, 1, 1]],  # L
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 0], [0, 1, 1]],  # Z
    ]

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
        
        try:
            self.font_m = pygame.font.SysFont("Consolas", 18)
            self.font_l = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_xl = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_m = pygame.font.Font(None, 22)
            self.font_l = pygame.font.Font(None, 28)
            self.font_xl = pygame.font.Font(None, 52)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        # State variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid.fill(0)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.current_piece = self._get_new_piece()
        self.next_piece = self._get_new_piece()
        
        self.base_drop_interval = 30  # Frames per grid cell drop (at 30fps = 1 second)
        self.drop_interval = self.base_drop_interval
        self.drop_timer = 0
        
        self.line_clear_animation = [] # Stores (row_index, timer)
        
        self.last_action_reward = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = -0.01 # Small penalty per step to encourage speed
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        placed_piece = False
        lines_cleared_this_step = 0

        # 1. Handle player actions
        if not self.line_clear_animation: # Freeze controls during line clear
            if space_held: # Hard drop
                # Sound: Hard drop
                while self._is_valid_position(self.current_piece, (1, 0)):
                    self.current_piece['y'] += 1
                self.drop_timer = self.drop_interval # Force lock
            else:
                # Rotations
                if shift_held: # CCW
                    self._rotate_piece(self.current_piece, -1)
                elif movement == 1: # CW (Up arrow)
                    self._rotate_piece(self.current_piece, 1)

                # Horizontal Movement
                if movement == 3: # Left
                    self._move_piece(-1)
                elif movement == 4: # Right
                    self._move_piece(1)

                # Soft Drop
                if movement == 2: # Down
                    self.drop_timer += 5 # Accelerate drop

        # 2. Update game logic (gravity)
        self.drop_timer += 1
        
        if self.drop_timer >= self.drop_interval:
            self.drop_timer = 0
            if self._is_valid_position(self.current_piece, (1, 0)):
                self.current_piece['y'] += 1
            else:
                # 3. Lock piece
                self._place_piece()
                placed_piece = True
                # Sound: Piece lock
                
                # 4. Check for line clears
                lines_cleared_this_step = self._check_and_start_clear_animation()
                if lines_cleared_this_step > 0:
                    # Sound: Line clear
                    self.lines_cleared += lines_cleared_this_step
                    self.score += self._calculate_line_clear_score(lines_cleared_this_step)
                
                # 5. Spawn next piece
                self.current_piece = self.next_piece
                self.next_piece = self._get_new_piece()
                
                # 6. Check for game over
                if not self._is_valid_position(self.current_piece):
                    self.game_over = True
                
                # 7. Update difficulty
                self.drop_interval = max(5, self.base_drop_interval - (self.lines_cleared // 20) * 3) # 0.1s faster per 20 lines

        # 8. Calculate reward
        reward += self._calculate_reward(lines_cleared_this_step, placed_piece, self.game_over)
        self.last_action_reward = reward
        self.score += reward

        # 9. Update line clear animation
        self._update_line_clear_animation()

        # 10. Check for termination
        terminated = self.game_over or self.lines_cleared >= 100 or self.steps >= 10000
        if self.lines_cleared >= 100 and not self.game_over:
            reward += 100
            self.score += 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over_screen("GAME OVER")
        elif self.lines_cleared >= 100:
            self._render_game_over_screen("YOU WIN!")

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
        }

    # --- Helper Methods ---

    def _get_new_piece(self):
        piece_idx = self.np_random.integers(0, len(self.PIECES))
        return {
            'shape': self.PIECES[piece_idx],
            'color_idx': piece_idx + 1,
            'x': self.GRID_WIDTH // 2 - len(self.PIECES[piece_idx][0]) // 2,
            'y': 0,
        }

    def _is_valid_position(self, piece, offset=(0, 0)):
        shape = piece['shape']
        off_x, off_y = piece['x'] + offset[1], piece['y'] + offset[0]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = off_x + c, off_y + r
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[y, x] == 0):
                        return False
        return True
    
    def _rotate_piece(self, piece, direction):
        # Sound: Rotate
        original_shape = piece['shape']
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*original_shape[::-1])]
        else: # Counter-clockwise
            new_shape = [list(row) for row in zip(*original_shape)][::-1]
        
        piece['shape'] = new_shape
        if not self._is_valid_position(piece):
            # Wall kick attempt
            if self._is_valid_position(piece, (0, 1)):
                piece['x'] += 1
            elif self._is_valid_position(piece, (0, -1)):
                piece['x'] -= 1
            else: # Revert if still invalid
                piece['shape'] = original_shape

    def _move_piece(self, dx):
        if self._is_valid_position(self.current_piece, (0, dx)):
            self.current_piece['x'] += dx
            # Sound: Move
            
    def _place_piece(self):
        shape = self.current_piece['shape']
        off_x, off_y = self.current_piece['x'], self.current_piece['y']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.grid[off_y + r, off_x + c] = self.current_piece['color_idx']

    def _check_and_start_clear_animation(self):
        full_lines = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                full_lines.append(r)
        
        if full_lines:
            self.line_clear_animation = [(row, 10) for row in full_lines] # 10 frames animation
        return len(full_lines)
    
    def _update_line_clear_animation(self):
        if not self.line_clear_animation:
            return
        
        new_animation = []
        for row_idx, timer in self.line_clear_animation:
            if timer > 1:
                new_animation.append((row_idx, timer - 1))
        self.line_clear_animation = new_animation
        
        if not self.line_clear_animation:
            self._finish_line_clear()

    def _finish_line_clear(self):
        rows_to_clear = sorted([item[0] for item in self.line_clear_animation if item[1] == 1], reverse=True)
        if not rows_to_clear:
            rows_to_clear = sorted(list(set(r for r, t in self.line_clear_animation)), reverse=True)
            # This is a fallback for the last frame
            if not self.line_clear_animation and hasattr(self, '_last_cleared_rows'):
                rows_to_clear = self._last_cleared_rows
        
        for row_idx in rows_to_clear:
            self.grid[1:row_idx + 1] = self.grid[0:row_idx]
            self.grid[0].fill(0)
        
        self._last_cleared_rows = rows_to_clear
        self.line_clear_animation = []

    def _calculate_line_clear_score(self, num_lines):
        if num_lines == 1: return 40
        if num_lines == 2: return 100
        if num_lines == 3: return 300
        if num_lines == 4: return 1200
        return 0
    
    def _calculate_reward(self, lines_cleared, placed_piece, game_over_loss):
        reward = 0
        if placed_piece:
            reward += 0.1
        
        reward_map = {1: 1, 2: 3, 3: 5, 4: 10}
        reward += reward_map.get(lines_cleared, 0)
        
        if game_over_loss:
            reward -= 50
            
        return reward

    def _render_game(self):
        # Draw grid background and border
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, grid_rect, 1)

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(c, r, self.grid[r, c])
        
        # Draw ghost piece
        if not self.game_over:
            ghost = self.current_piece.copy()
            while self._is_valid_position(ghost, (1, 0)):
                ghost['y'] += 1
            self._draw_piece(ghost, is_ghost=True)

        # Draw current piece
        if not self.game_over:
            self._draw_piece(self.current_piece)

        # Draw line clear animation
        for row_idx, timer in self.line_clear_animation:
            y = self.GRID_Y_OFFSET + row_idx * self.CELL_SIZE
            alpha = 255 - (abs(5 - timer) * 50)
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_WHITE, alpha))
            self.screen.blit(flash_surface, (self.GRID_X_OFFSET, y))

    def _draw_piece(self, piece, is_ghost=False):
        shape = piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(piece['x'] + c, piece['y'] + r, piece['color_idx'], is_ghost)
    
    def _draw_block(self, grid_c, grid_r, color_idx, is_ghost=False):
        x = self.GRID_X_OFFSET + grid_c * self.CELL_SIZE
        y = self.GRID_Y_OFFSET + grid_r * self.CELL_SIZE
        color = self.PIECE_COLORS[color_idx - 1]
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=3)
        else:
            light_color = tuple(min(255, c + 60) for c in color)
            dark_color = tuple(max(0, c - 60) for c in color)
            
            pygame.draw.rect(self.screen, light_color, rect.move(-1, -1), border_radius=4)
            pygame.draw.rect(self.screen, dark_color, rect.move(1, 1), border_radius=4)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            pygame.draw.rect(self.screen, self.COLOR_BG, rect.inflate(-6, -6))

    def _render_ui(self):
        # Score
        self._draw_text("SCORE", self.font_l, self.COLOR_UI_HEADER, 50, 30)
        self._draw_text(f"{int(self.score):,}", self.font_m, self.COLOR_UI_TEXT, 50, 60)
        
        # Lines
        self._draw_text("LINES", self.font_l, self.COLOR_UI_HEADER, 50, 110)
        self._draw_text(f"{self.lines_cleared}", self.font_m, self.COLOR_UI_TEXT, 50, 140)

        # Next Piece
        self._draw_text("NEXT", self.font_l, self.COLOR_UI_HEADER, 520, 30)
        next_piece_copy = self.next_piece.copy()
        next_piece_copy['x'] = (self.SCREEN_WIDTH - self.GRID_X_OFFSET - self.CELL_SIZE * 4) // (2 * self.CELL_SIZE) + self.GRID_WIDTH + 3
        next_piece_copy['y'] = 3
        self._draw_piece(next_piece_copy)

    def _render_game_over_screen(self, message):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        self._draw_text(message, self.font_xl, self.COLOR_WHITE, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, center=True)

    def _draw_text(self, text, font, color, x, y, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(text_surface, text_rect)

    def validate_implementation(self):
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gymnasium Block Dropper")
    clock = pygame.time.Clock()

    action = env.action_space.sample()
    action.fill(0)

    print("--- Playing Game ---")
    print(env.user_guide)

    while not done:
        # Human input mapping
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0

        # Buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render to the display
        draw_surface = pygame.transform.flip(pygame.surfarray.make_surface(obs), False, True)
        render_screen.blit(draw_surface, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']:.2f}, Lines: {info['lines_cleared']}, Steps: {info['steps']}")
    pygame.quit()