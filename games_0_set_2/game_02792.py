
# Generated: 2025-08-27T21:26:35.676457
# Source Brief: brief_02792.md
# Brief Index: 2792

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ for soft drop. "
        "Hold shift to rotate counter-clockwise and press space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down falling block puzzle game. "
        "Strategically rotate and place blocks to clear lines and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (30, 35, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_GHOST = (255, 255, 255, 40)

    TETROMINO_COLORS = [
        (40, 40, 40),      # 0: Empty
        (0, 240, 240),     # I (Cyan)
        (0, 0, 240),       # J (Blue)
        (240, 160, 0),     # L (Orange)
        (240, 240, 0),     # O (Yellow)
        (0, 240, 0),       # S (Green)
        (160, 0, 240),     # T (Purple)
        (240, 0, 0),       # Z (Red)
    ]

    # Tetromino shapes
    TETROMINOES = {
        'I': [[1, 1, 1, 1]],
        'J': [[1, 0, 0], [1, 1, 1]],
        'L': [[0, 0, 1], [1, 1, 1]],
        'O': [[1, 1], [1, 1]],
        'S': [[0, 1, 1], [1, 1, 0]],
        'T': [[0, 1, 0], [1, 1, 1]],
        'Z': [[1, 1, 0], [0, 1, 1]]
    }

    # Playfield dimensions
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    CELL_SIZE = 18
    BORDER_WIDTH = 4

    # Game parameters
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 1000
    FALL_SPEED_NORMAL = 15  # Ticks per grid cell fall
    FALL_SPEED_SOFT_DROP = 2 # Ticks per grid cell fall
    ACTION_COOLDOWN = 5 # Frames
    MOVE_COOLDOWN_INITIAL = 10 # Frames
    MOVE_COOLDOWN_REPEAT = 3 # Frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400

        self.playfield_width_px = self.GRID_WIDTH * self.CELL_SIZE
        self.playfield_height_px = self.GRID_HEIGHT * self.CELL_SIZE
        self.playfield_x = (self.screen_width - self.playfield_width_px) // 2
        self.playfield_y = (self.screen_height - self.playfield_height_px) // 2

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables
        self.board = None
        self.current_piece = None
        self.piece_pos = None
        self.piece_rot = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.line_clear_effects = []

        self.action_cooldowns = {}
        self.last_move_direction = 0

        self.reset()
        
        # self.validate_implementation() # Optional: run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.line_clear_effects = []
        
        self.action_cooldowns = {
            'rotate_cw': 0, 'rotate_ccw': 0, 'hard_drop': 0,
            'move': 0
        }
        self.last_move_direction = 0

        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed

        if not self.game_over:
            self._handle_action(action)
            self._update_game_state()
            reward += self._check_line_clears()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cooldowns ---
        for key in self.action_cooldowns:
            if self.action_cooldowns[key] > 0:
                self.action_cooldowns[key] -= 1

        # --- Actions ---
        # Hard Drop (Space)
        if space_held and self.action_cooldowns['hard_drop'] == 0:
            self._hard_drop()
            self.action_cooldowns['hard_drop'] = self.ACTION_COOLDOWN * 2
            # Sound: Hard drop thud

        # Rotation CW (Up)
        if movement == 1 and self.action_cooldowns['rotate_cw'] == 0:
            self._rotate_piece(clockwise=True)
            self.action_cooldowns['rotate_cw'] = self.ACTION_COOLDOWN
            # Sound: Rotation click

        # Rotation CCW (Shift)
        if shift_held and self.action_cooldowns['rotate_ccw'] == 0:
            self._rotate_piece(clockwise=False)
            self.action_cooldowns['rotate_ccw'] = self.ACTION_COOLDOWN
            # Sound: Rotation click

        # Horizontal Movement (Left/Right)
        move_dir = 0
        if movement == 3: move_dir = -1
        if movement == 4: move_dir = 1
        
        if move_dir != 0:
            if self.action_cooldowns['move'] == 0:
                self._move_piece(dx=move_dir, dy=0)
                # Set appropriate cooldown based on if direction changed
                if move_dir == self.last_move_direction:
                    self.action_cooldowns['move'] = self.MOVE_COOLDOWN_REPEAT
                else:
                    self.action_cooldowns['move'] = self.MOVE_COOLDOWN_INITIAL
        self.last_move_direction = move_dir


    def _update_game_state(self):
        # Piece falling
        self.fall_timer += 1
        
        soft_drop = (self.action_space.sample()[0] == 2) # Get soft drop from current (unused) action
        fall_speed = self.FALL_SPEED_SOFT_DROP if soft_drop else self.FALL_SPEED_NORMAL

        if self.fall_timer >= fall_speed:
            self.fall_timer = 0
            if not self._move_piece(dx=0, dy=1):
                self._lock_piece()
                self._spawn_new_piece()

        # Update line clear effects
        self.line_clear_effects = [
            (x, y, life - 1) for x, y, life in self.line_clear_effects if life > 0
        ]

    def _spawn_new_piece(self):
        piece_type = random.choice(list(self.TETROMINOES.keys()))
        self.current_piece = {
            'shape': np.array(self.TETROMINOES[piece_type]),
            'color_idx': list(self.TETROMINOES.keys()).index(piece_type) + 1
        }
        self.piece_rot = 0
        start_x = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
        self.piece_pos = [start_x, 0]

        if not self._is_valid_position(self.current_piece['shape'], self.piece_pos):
            self.game_over = True

    def _is_valid_position(self, shape, pos):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    board_x, board_y = pos[0] + c, pos[1] + r
                    if not (0 <= board_x < self.GRID_WIDTH and 0 <= board_y < self.GRID_HEIGHT):
                        return False  # Out of bounds
                    if self.board[board_y, board_x] != 0:
                        return False  # Collision with another piece
        return True

    def _move_piece(self, dx, dy):
        new_pos = [self.piece_pos[0] + dx, self.piece_pos[1] + dy]
        if self._is_valid_position(self.current_piece['shape'], new_pos):
            self.piece_pos = new_pos
            # Sound: Move tick
            return True
        return False

    def _rotate_piece(self, clockwise=True):
        original_shape = self.current_piece['shape']
        if clockwise:
            new_shape = np.rot90(original_shape, k=-1)
        else:
            new_shape = np.rot90(original_shape, k=1)

        # Wall kick checks
        offsets_to_test = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        for ox, oy in offsets_to_test:
            test_pos = [self.piece_pos[0] + ox, self.piece_pos[1] + oy]
            if self._is_valid_position(new_shape, test_pos):
                self.current_piece['shape'] = new_shape
                self.piece_pos = test_pos
                return True
        return False

    def _hard_drop(self):
        while self._move_piece(0, 1):
            pass
        self._lock_piece()
        self._spawn_new_piece()

    def _lock_piece(self):
        shape = self.current_piece['shape']
        pos = self.piece_pos
        color = self.current_piece['color_idx']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.board[pos[1] + r, pos[0] + c] = color
        # Sound: Piece lock
        self.current_piece = None

    def _check_line_clears(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.board[r, :] != 0):
                lines_to_clear.append(r)

        if lines_to_clear:
            # Sound: Line clear
            for r in lines_to_clear:
                # Add particles for visual effect
                for c in range(self.GRID_WIDTH):
                    self.line_clear_effects.append((c, r, 10)) # 10 frames life
            
            # Remove cleared lines and shift down
            self.board = np.delete(self.board, lines_to_clear, axis=0)
            new_rows = np.zeros((len(lines_to_clear), self.GRID_WIDTH), dtype=int)
            self.board = np.vstack((new_rows, self.board))

            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            
            # Scoring: simple +1 per line as per brief
            self.score += num_cleared
            return float(num_cleared)
        return 0

    def _check_termination(self):
        if self.game_over:
            return True
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }

    def _render_game(self):
        # Draw playfield border and background
        border_rect = pygame.Rect(
            self.playfield_x - self.BORDER_WIDTH,
            self.playfield_y - self.BORDER_WIDTH,
            self.playfield_width_px + self.BORDER_WIDTH * 2,
            self.playfield_height_px + self.BORDER_WIDTH * 2
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID, border_rect, border_radius=5)
        playfield_rect = pygame.Rect(
            self.playfield_x, self.playfield_y, self.playfield_width_px, self.playfield_height_px
        )
        pygame.draw.rect(self.screen, self.COLOR_BG, playfield_rect)
        
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.playfield_x + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.playfield_y), (px, self.playfield_y + self.playfield_height_px))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.playfield_y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.playfield_x, py), (self.playfield_x + self.playfield_width_px, py))

        # Draw locked blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.board[r, c] != 0:
                    self._draw_cell(c, r, self.TETROMINO_COLORS[self.board[r, c]])

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_pos = list(self.piece_pos)
            while self._is_valid_position(self.current_piece['shape'], [ghost_pos[0], ghost_pos[1] + 1]):
                ghost_pos[1] += 1
            self._draw_piece(self.current_piece, ghost_pos, self.COLOR_GHOST, is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            color = self.TETROMINO_COLORS[self.current_piece['color_idx']]
            self._draw_piece(self.current_piece, self.piece_pos, color)
        
        # Draw line clear effects
        for x, y, life in self.line_clear_effects:
            alpha = int(255 * (life / 10))
            px = self.playfield_x + x * self.CELL_SIZE
            py = self.playfield_y + y * self.CELL_SIZE
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((255, 255, 255, alpha))
            self.screen.blit(s, (px, py))

    def _draw_piece(self, piece, pos, color, is_ghost=False):
        shape = piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(pos[0] + c, pos[1] + r, color, is_ghost)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        px = self.playfield_x + grid_x * self.CELL_SIZE
        py = self.playfield_y + grid_y * self.CELL_SIZE
        
        cell_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)

        if is_ghost:
            pygame.draw.rect(self.screen, color, cell_rect)
        else:
            highlight_color = tuple(min(255, c + 50) for c in color)
            shadow_color = tuple(max(0, c - 50) for c in color)
            
            pygame.draw.rect(self.screen, color, cell_rect)
            pygame.draw.line(self.screen, highlight_color, (px, py), (px + self.CELL_SIZE - 1, py))
            pygame.draw.line(self.screen, highlight_color, (px, py), (px, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow_color, (px + self.CELL_SIZE - 1, py), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow_color, (px, py + self.CELL_SIZE - 1), (px + self.CELL_SIZE - 1, py + self.CELL_SIZE - 1))


    def _render_ui(self):
        # Score
        score_text = self.font_ui.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_ui.render(f"{self.score:06d}", True, self.COLOR_UI_VALUE)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(score_val, (20, 45))
        
        # Lines
        lines_text = self.font_ui.render("LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_ui.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_UI_VALUE)
        self.screen.blit(lines_text, (self.screen_width - lines_text.get_width() - 20, 20))
        self.screen.blit(lines_val, (self.screen_width - lines_val.get_width() - 20, 45))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
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
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Interactive Human Play ---
    # This part is for demonstration and debugging.
    # It allows a human to play the game.
    
    env.reset()
    running = True
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, released, released
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()

        # Map keyboard state to MultiDiscrete action
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            env.reset()
            
        clock.tick(30) # Run at 30 FPS for human play

    env.close()