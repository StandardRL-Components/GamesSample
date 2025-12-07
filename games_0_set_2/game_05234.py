
# Generated: 2025-08-28T04:23:47.160053
# Source Brief: brief_05234.md
# Brief Index: 5234

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, Shift to rotate counter-clockwise, "
        "↓ for soft drop, Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, falling block puzzle game. Clear lines to score points, "
        "but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    CELL_SIZE = 18
    
    GRID_X = (SCREEN_WIDTH - PLAYFIELD_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - PLAYFIELD_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 60)
    COLOR_UI_BG = (30, 30, 45)
    COLOR_TEXT = (220, 220, 240)
    COLOR_GHOST = (255, 255, 255, 50)
    
    # Tetromino shapes and colors
    TETROMINOES = {
        'I': ([[1, 1, 1, 1]], (60, 200, 200)),
        'O': ([[1, 1], [1, 1]], (240, 240, 80)),
        'T': ([[0, 1, 0], [1, 1, 1]], (160, 80, 240)),
        'S': ([[0, 1, 1], [1, 1, 0]], (80, 240, 80)),
        'Z': ([[1, 1, 0], [0, 1, 1]], (240, 80, 80)),
        'J': ([[1, 0, 0], [1, 1, 1]], (80, 80, 240)),
        'L': ([[0, 0, 1], [1, 1, 1]], (240, 160, 80)),
    }
    
    SHAPES = list(TETROMINOES.keys())

    # Game settings
    MAX_STEPS = 5000
    WIN_SCORE = 1000
    INITIAL_DROP_SPEED = 0.5  # seconds per grid cell
    SPEED_INCREASE_INTERVAL = 200
    SPEED_INCREASE_AMOUNT = 0.01
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 24)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        
        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.won = False
        
        self.drop_speed = self.INITIAL_DROP_SPEED
        self.drop_timer = 0
        
        self.prev_action = np.array([0, 0, 0])
        
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        
        self.reset()
        self.validate_implementation()
    
    def _create_piece(self):
        shape_key = self.np_random.choice(self.SHAPES)
        shape = self.TETROMINOES[shape_key][0]
        return {
            "shape_key": shape_key,
            "shape": shape,
            "color": self.TETROMINOES[shape_key][1],
            "x": self.PLAYFIELD_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0,
        }

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        for r, row in enumerate(piece["shape"]):
            for c, cell in enumerate(row):
                if cell:
                    x = piece["x"] + c + offset_x
                    y = piece["y"] + r + offset_y
                    if not (0 <= x < self.PLAYFIELD_WIDTH and 0 <= y < self.PLAYFIELD_HEIGHT and self.grid[y][x] == 0):
                        return False
        return True

    def _rotate_piece(self, piece, clockwise=True):
        shape = piece["shape"]
        if clockwise:
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else:
            new_shape = [list(row) for row in zip(*shape)][::-1]
        
        rotated_piece = piece.copy()
        rotated_piece["shape"] = new_shape
        
        # Basic wall kick
        if not self._is_valid_position(rotated_piece):
            # Try moving right
            if self._is_valid_position(rotated_piece, offset_x=1):
                rotated_piece["x"] += 1
                return rotated_piece
            # Try moving left
            if self._is_valid_position(rotated_piece, offset_x=-1):
                rotated_piece["x"] -= 1
                return rotated_piece
            # Try moving further right for I-piece
            if piece['shape_key'] == 'I' and self._is_valid_position(rotated_piece, offset_x=2):
                 rotated_piece["x"] += 2
                 return rotated_piece
            # Try moving further left for I-piece
            if piece['shape_key'] == 'I' and self._is_valid_position(rotated_piece, offset_x=-2):
                 rotated_piece["x"] -= 2
                 return rotated_piece
            return None  # Rotation failed
        return rotated_piece

    def _lock_piece(self):
        # sfx: block_lock.wav
        piece = self.current_piece
        for r, row in enumerate(piece["shape"]):
            for c, cell in enumerate(row):
                if cell:
                    x, y = piece["x"] + c, piece["y"] + r
                    if 0 <= y < self.PLAYFIELD_HEIGHT:
                        self.grid[y][x] = self.TETROMINOES[piece["shape_key"]][1]
        
        self.current_piece = self.next_piece
        self.next_piece = self._create_piece()
        
        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            # sfx: game_over.wav
        
        return self._check_and_clear_lines()

    def _check_and_clear_lines(self):
        lines_cleared = 0
        full_rows = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if all(self.grid[r]):
                full_rows.append(r)
        
        if full_rows:
            # sfx: line_clear.wav
            self.lines_to_clear = full_rows
            self.clear_animation_timer = 5 # frames
            lines_cleared = len(full_rows)
            
            # Score bonus for multiple lines
            score_map = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += score_map.get(lines_cleared, 0)

        return lines_cleared

    def _finalize_line_clear(self):
        for r in sorted(self.lines_to_clear, reverse=True):
            del self.grid[r]
            self.grid.insert(0, [0] * self.PLAYFIELD_WIDTH)
        self.lines_to_clear = []

    def _calculate_placement_penalty(self):
        penalty = 0
        piece = self.current_piece
        for r, row in enumerate(piece["shape"]):
            for c, cell in enumerate(row):
                if cell:
                    x, y = piece["x"] + c, piece["y"] + r
                    # Check for creating a hole directly underneath
                    if y + 1 < self.PLAYFIELD_HEIGHT and self.grid[y + 1][x] == 0:
                        is_covered = False
                        for r2 in range(y + 1, self.PLAYFIELD_HEIGHT):
                            if self.grid[r2][x] != 0:
                                is_covered = True
                                break
                        if not is_covered:
                             penalty += 1
        return penalty

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = [[0] * self.PLAYFIELD_WIDTH for _ in range(self.PLAYFIELD_HEIGHT)]
        self.current_piece = self._create_piece()
        self.next_piece = self._create_piece()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.won = False
        
        self.drop_speed = self.INITIAL_DROP_SPEED
        self.drop_timer = 0
        self.prev_action = np.array([0, 0, 0])
        
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if self.game_over or self.won:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Handle line clear animation state
        if self.clear_animation_timer > 0:
            self.clear_animation_timer -= 1
            if self.clear_animation_timer == 0:
                self._finalize_line_clear()
            # No other actions while clearing lines
            return self._get_observation(), 0, False, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        prev_movement, prev_space, prev_shift = self.prev_action[0], self.prev_action[1] == 1, self.prev_action[2] == 1
        self.prev_action = action

        # Handle single-press actions
        if movement == 1 and prev_movement != 1:  # Up / Rotate CW
            # sfx: rotate.wav
            rotated = self._rotate_piece(self.current_piece, clockwise=True)
            if rotated: self.current_piece = rotated
        if shift_held and not prev_shift:  # Shift / Rotate CCW
            # sfx: rotate.wav
            rotated = self._rotate_piece(self.current_piece, clockwise=False)
            if rotated: self.current_piece = rotated
        if space_held and not prev_space:  # Space / Hard Drop
            # sfx: hard_drop.wav
            rows_dropped = 0
            while self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
                rows_dropped += 1
            reward += rows_dropped * 0.05 # Small reward for decisive drops
            penalty = self._calculate_placement_penalty()
            lines_cleared = self._lock_piece()
            reward += lines_cleared * 10 - penalty

        # Handle continuous actions
        if movement == 3:  # Left
            if self._is_valid_position(self.current_piece, offset_x=-1):
                self.current_piece["x"] -= 1
                reward -= 0.01
        elif movement == 4:  # Right
            if self._is_valid_position(self.current_piece, offset_x=1):
                self.current_piece["x"] += 1
                reward -= 0.01
        elif movement == 2:  # Down / Soft Drop
            if self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
                reward += 0.1
                self.drop_timer = 0 # Reset auto-drop timer
            else:
                penalty = self._calculate_placement_penalty()
                lines_cleared = self._lock_piece()
                reward += lines_cleared * 10 - penalty

        # Auto-drop logic
        self.drop_timer += self.clock.get_time() / 1000.0
        if self.drop_timer >= self.drop_speed:
            self.drop_timer = 0
            if self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
            else:
                penalty = self._calculate_placement_penalty()
                lines_cleared = self._lock_piece()
                reward += lines_cleared * 10 - penalty
        
        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.drop_speed = max(0.05, self.drop_speed - self.SPEED_INCREASE_AMOUNT)

        # Termination checks
        terminated = self._check_termination()
        if terminated:
            if self.won:
                reward += 100
            elif self.game_over:
                reward -= 100
        
        self.clock.tick(30)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.score >= self.WIN_SCORE and not self.won:
            self.won = True
            self.game_over = True # End game on win
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.GRID_X, self.GRID_Y, self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.PLAYFIELD_HEIGHT * self.CELL_SIZE))
        
        # Draw grid lines
        for x in range(self.PLAYFIELD_WIDTH + 1):
            px = self.GRID_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y), (px, self.GRID_Y + self.PLAYFIELD_HEIGHT * self.CELL_SIZE))
        for y in range(self.PLAYFIELD_HEIGHT + 1):
            py = self.GRID_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, py), (self.GRID_X + self.PLAYFIELD_WIDTH * self.CELL_SIZE, py))

        # Draw locked pieces
        for r, row in enumerate(self.grid):
            for c, color in enumerate(row):
                if color != 0:
                    self._draw_block(c, r, color)
        
        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost = self.current_piece.copy()
            while self._is_valid_position(ghost, offset_y=1):
                ghost["y"] += 1
            self._draw_block(ghost["x"], ghost["y"], self.COLOR_GHOST, ghost["shape"], is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            self._draw_block(self.current_piece["x"], self.current_piece["y"], self.current_piece["color"], self.current_piece["shape"])

        # Draw line clear animation
        if self.clear_animation_timer > 0:
            flash_color = (255, 255, 255) if self.clear_animation_timer % 2 == 0 else self.COLOR_UI_BG
            for r in self.lines_to_clear:
                pygame.draw.rect(self.screen, flash_color, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE, self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.CELL_SIZE))

    def _draw_block(self, grid_x, grid_y, color, shape=None, is_ghost=False):
        if shape is None: # Single block
            px, py = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
            main_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            darker_color = (max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, tuple(darker_color), main_rect)
            inner_rect = main_rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect)
        else: # Full piece
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px, py = self.GRID_X + (grid_x + c) * self.CELL_SIZE, self.GRID_Y + (grid_y + r) * self.CELL_SIZE
                        main_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                        
                        if is_ghost:
                            pygame.draw.rect(self.screen, color, main_rect, 2, border_radius=2)
                        else:
                            darker_color = tuple(max(0, val - 50) for val in color)
                            pygame.draw.rect(self.screen, darker_color, main_rect, border_radius=3)
                            inner_rect = main_rect.inflate(-4, -4)
                            pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)

    def _render_ui(self):
        # UI panel for score
        score_panel_x = self.GRID_X - 160
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (score_panel_x, self.GRID_Y, 150, 80), border_radius=10)
        score_text = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (score_panel_x + 20, self.GRID_Y + 15))
        score_val = self.font_medium.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (score_panel_x + 20, self.GRID_Y + 40))
        
        # UI panel for next piece
        next_panel_x = self.GRID_X + self.PLAYFIELD_WIDTH * self.CELL_SIZE + 10
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (next_panel_x, self.GRID_Y, 120, 120), border_radius=10)
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (next_panel_x + 35, self.GRID_Y + 15))
        if self.next_piece:
            shape = self.next_piece["shape"]
            color = self.next_piece["color"]
            w = len(shape[0])
            h = len(shape)
            start_x = next_panel_x + (120 - w * self.CELL_SIZE) // 2
            start_y = self.GRID_Y + (120 - h * self.CELL_SIZE) // 2 + 10
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px, py = start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE
                        main_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                        darker_color = tuple(max(0, val - 50) for val in color)
                        pygame.draw.rect(self.screen, darker_color, main_rect, border_radius=3)
                        inner_rect = main_rect.inflate(-4, -4)
                        pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.won else "GAME OVER"
            text = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": (len(self.lines_to_clear) if self.clear_animation_timer > 0 else 0),
            "drop_speed": self.drop_speed,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game directly
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Change to 'windows' or 'mac' or remove if not on linux
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a real screen for human play
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gymnasium Tetris")
    
    terminated = False
    
    # Mapping from Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: (1, 0, 0),
        pygame.K_DOWN: (2, 0, 0),
        pygame.K_LEFT: (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
        pygame.K_LSHIFT: (0, 0, 1),
        pygame.K_RSHIFT: (0, 0, 1),
    }

    action = np.array([0, 0, 0]) # No-op
    
    while not terminated:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Reset action on key up/down to handle presses correctly
            if event.type in (pygame.KEYUP, pygame.KEYDOWN):
                action = np.array([0, 0, 0])
                keys = pygame.key.get_pressed()
                
                # Combine inputs
                if keys[pygame.K_UP]: action[0] = 1
                elif keys[pygame.K_DOWN]: action[0] = 2
                elif keys[pygame.K_LEFT]: action[0] = 3
                elif keys[pygame.K_RIGHT]: action[0] = 4
                else: action[0] = 0
                
                if keys[pygame.K_SPACE]: action[1] = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the real screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds before closing
            
    env.close()