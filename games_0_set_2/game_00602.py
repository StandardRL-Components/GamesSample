
# Generated: 2025-08-27T14:09:17.219872
# Source Brief: brief_00602.md
# Brief Index: 602

        
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
    """
    A minimalist, grid-based puzzle game where the player places falling blocks
    (tetrominoes) to clear lines and achieve a high score. The game prioritizes
    visual clarity, smooth animations, and satisfying gameplay feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. "
        "Space to hard drop. Shift to hold/swap piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically place falling blocks to clear lines and achieve a high score "
        "in this minimalist, grid-based puzzle game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 10, 20
        self.CELL_SIZE = 19
        self.BOARD_PIXEL_WIDTH = self.BOARD_WIDTH * self.CELL_SIZE
        self.BOARD_PIXEL_HEIGHT = self.BOARD_HEIGHT * self.CELL_SIZE
        self.BOARD_X_OFFSET = (self.WIDTH - self.BOARD_PIXEL_WIDTH) // 2
        self.BOARD_Y_OFFSET = (self.HEIGHT - self.BOARD_PIXEL_HEIGHT) // 2
        self.MAX_STEPS = 10000
        self.WIN_SCORE = 500

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_BG = (30, 30, 45)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255)
        self.TETROMINO_COLORS = [
            (0, 240, 240),  # I (Cyan)
            (240, 240, 0),  # O (Yellow)
            (160, 0, 240),  # T (Purple)
            (0, 0, 240),    # J (Blue)
            (240, 160, 0),  # L (Orange)
            (0, 240, 0),    # S (Green)
            (240, 0, 0),    # Z (Red)
        ]

        # Tetromino shapes (indices correspond to colors)
        self.TETROMINOES = [
            [[1, 1, 1, 1]],  # I
            [[1, 1], [1, 1]],  # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]],  # L
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 1, 0], [0, 1, 1]],  # Z
        ]
        
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
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # Initialize state variables
        self.reset()

        # Run self-check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.fall_speed = 0.2
        self.fall_counter = 0.0
        
        self.piece_bag = list(range(len(self.TETROMINOES)))
        random.shuffle(self.piece_bag)
        
        self._spawn_piece() # Spawns current_piece
        self._spawn_piece() # Spawns next_piece
        
        self.held_piece_type = None
        self.can_hold = True
        
        self.flash_timer = 0
        self.cleared_lines_indices = []

        self.last_reward_info = {}

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        placed_info = None
        lines_cleared = 0

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- Action Phase ---
        if shift_held and self.can_hold:
            self._hold_piece()
        
        if movement == 1: self._rotate() # Up
        if movement == 3: self._move(-1) # Left
        if movement == 4: self._move(1)  # Right
        
        soft_drop = movement == 2
        
        # --- Physics and Game Logic Phase ---
        if space_held: # Hard drop
            placed_info = self._hard_drop()
            # sfx: hard_drop_sound
        else:
            # Gravity
            self.fall_counter += self.fall_speed
            if soft_drop:
                self.fall_counter += 0.5 # Soft drop bonus speed
            
            if self.fall_counter >= 1.0:
                moves = int(self.fall_counter)
                self.fall_counter %= 1.0
                for _ in range(moves):
                    if not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'] + 1)):
                        self.current_piece['y'] += 1
                    else:
                        placed_info = self._place_piece()
                        # sfx: piece_land_sound
                        break
        
        # --- Post-Placement Phase ---
        if placed_info:
            lines_cleared = self._clear_lines()
            if lines_cleared > 0:
                self.flash_timer = 5 # Flash for 5 frames
                # sfx: line_clear_sound
            
            self._spawn_piece()
            self.can_hold = True # Allow holding again after placing a piece
            
            if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
                self.game_over = True
        
        # --- Update and Reward Phase ---
        reward = self._calculate_reward(placed_info, lines_cleared)
        self.score += self._calculate_score(lines_cleared)

        # Difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.fall_speed = min(1.0, self.fall_speed + 0.01)

        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_reward(self, placed_info, lines_cleared):
        reward = 0
        if self.game_over:
            return -50.0

        if self.score >= self.WIN_SCORE:
            return 100.0

        if placed_info:
            reward += 0.1 # Reward for placing a block
            
            # Penalty for empty cells underneath
            holes = 0
            piece_shape = placed_info['shape']
            px, py = placed_info['x'], placed_info['y']
            for r in range(len(piece_shape)):
                for c in range(len(piece_shape[r])):
                    if piece_shape[r][c] == 1:
                        for y_check in range(py + r + 1, self.BOARD_HEIGHT):
                            if self.board[y_check][px + c] == 0:
                                holes += 1
            reward -= 0.01 * holes

        if lines_cleared == 1: reward += 1
        elif lines_cleared > 1: reward += 5 # Bonus for multi-line clear

        return reward
    
    def _calculate_score(self, lines_cleared):
        if lines_cleared == 1: return 10
        if lines_cleared == 2: return 30
        if lines_cleared == 3: return 60
        if lines_cleared == 4: return 100 # "Tetris"
        return 0

    def _spawn_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.TETROMINOES)))
            random.shuffle(self.piece_bag)
            
        piece_type = self.piece_bag.pop(0)
        shape = self.TETROMINOES[piece_type]
        
        self.current_piece = self.next_piece if hasattr(self, 'next_piece') else None
        
        self.next_piece = {
            'type': piece_type,
            'shape': shape,
            'x': self.BOARD_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0
        }
        
    def _hold_piece(self):
        # sfx: hold_sound
        if self.held_piece_type is None:
            self.held_piece_type = self.current_piece['type']
            self._spawn_piece()
        else:
            held_type = self.held_piece_type
            self.held_piece_type = self.current_piece['type']
            
            shape = self.TETROMINOES[held_type]
            self.current_piece = {
                'type': held_type,
                'shape': shape,
                'x': self.BOARD_WIDTH // 2 - len(shape[0]) // 2,
                'y': 0
            }
        self.can_hold = False

    def _check_collision(self, shape, offset):
        off_x, off_y = offset
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    board_x, board_y = c + off_x, r + off_y
                    if not (0 <= board_x < self.BOARD_WIDTH and 0 <= board_y < self.BOARD_HEIGHT and self.board[board_y][board_x] == 0):
                        return True
        return False

    def _rotate(self):
        if self.current_piece is None: return
        # sfx: rotate_sound
        original_shape = self.current_piece['shape']
        rotated_shape = [list(row) for row in zip(*original_shape[::-1])]
        
        if not self._check_collision(rotated_shape, (self.current_piece['x'], self.current_piece['y'])):
            self.current_piece['shape'] = rotated_shape
        # Basic wall kick (try moving one space left/right)
        elif not self._check_collision(rotated_shape, (self.current_piece['x'] + 1, self.current_piece['y'])):
            self.current_piece['x'] += 1
            self.current_piece['shape'] = rotated_shape
        elif not self._check_collision(rotated_shape, (self.current_piece['x'] - 1, self.current_piece['y'])):
            self.current_piece['x'] -= 1
            self.current_piece['shape'] = rotated_shape

    def _move(self, dx):
        if self.current_piece is None: return
        # sfx: move_sound
        if not self._check_collision(self.current_piece['shape'], (self.current_piece['x'] + dx, self.current_piece['y'])):
            self.current_piece['x'] += dx

    def _hard_drop(self):
        if self.current_piece is None: return None
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], y + 1)):
            y += 1
        self.current_piece['y'] = y
        return self._place_piece()

    def _place_piece(self):
        if self.current_piece is None: return None
        shape = self.current_piece['shape']
        px, py = self.current_piece['x'], self.current_piece['y']
        color_idx = self.current_piece['type'] + 1
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.board[py + r][px + c] = color_idx
        
        placed_info = self.current_piece.copy()
        self.current_piece = None
        return placed_info

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.board) if np.all(row > 0)]
        
        if not lines_to_clear:
            return 0
        
        self.cleared_lines_indices = lines_to_clear
        
        # Remove lines from bottom up
        for r in sorted(lines_to_clear, reverse=True):
            self.board = np.delete(self.board, r, axis=0)
            
        # Add new empty lines at the top
        new_lines = np.zeros((len(lines_to_clear), self.BOARD_WIDTH), dtype=int)
        self.board = np.vstack((new_lines, self.board))
        
        return len(lines_to_clear)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.flash_timer > 0:
            self._render_line_clear_flash()
            self.flash_timer -= 1
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and background
        board_rect = pygame.Rect(self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET, self.BOARD_PIXEL_WIDTH, self.BOARD_PIXEL_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, board_rect)
        for i in range(self.BOARD_WIDTH + 1):
            x = self.BOARD_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_Y_OFFSET), (x, self.BOARD_Y_OFFSET + self.BOARD_PIXEL_HEIGHT))
        for i in range(self.BOARD_HEIGHT + 1):
            y = self.BOARD_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, y), (self.BOARD_X_OFFSET + self.BOARD_PIXEL_WIDTH, y))

        # Draw placed blocks
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                if self.board[r][c] > 0:
                    color = self.TETROMINO_COLORS[self.board[r][c] - 1]
                    self._draw_cell(c, r, color)
        
        # Draw ghost and falling piece
        if self.current_piece:
            self._render_ghost_piece()
            self._draw_piece(
                self.current_piece['shape'],
                (self.current_piece['x'], self.current_piece['y']),
                self.TETROMINO_COLORS[self.current_piece['type']]
            )

    def _render_ghost_piece(self):
        if self.current_piece is None: return
        ghost_y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], ghost_y + 1)):
            ghost_y += 1
        
        self._draw_piece(
            self.current_piece['shape'],
            (self.current_piece['x'], ghost_y),
            self.COLOR_GHOST,
            is_ghost=True
        )
    
    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        score_label = self.font_small.render("SCORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_label, (20, 60))

        # Next Piece
        next_label = self.font_medium.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_label, (self.WIDTH - 150, 20))
        if self.next_piece:
            self._draw_preview_piece(self.next_piece, (self.WIDTH - 120, 70))
            
        # Held Piece
        hold_label = self.font_medium.render("HOLD", True, self.COLOR_UI_TEXT)
        self.screen.blit(hold_label, (self.WIDTH - 150, 200))
        if self.held_piece_type is not None:
            held_piece_preview = {
                'type': self.held_piece_type,
                'shape': self.TETROMINOES[self.held_piece_type]
            }
            self._draw_preview_piece(held_piece_preview, (self.WIDTH - 120, 250))
            
    def _draw_cell(self, board_c, board_r, color, is_ghost=False):
        x = self.BOARD_X_OFFSET + board_c * self.CELL_SIZE
        y = self.BOARD_Y_OFFSET + board_r * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
        else:
            pygame.draw.rect(self.screen, color, rect)
            # Bevel effect for depth
            light_color = tuple(min(255, c + 40) for c in color)
            dark_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, light_color, (x, y), (x + self.CELL_SIZE - 1, y), 2)
            pygame.draw.line(self.screen, light_color, (x, y), (x, y + self.CELL_SIZE - 1), 2)
            pygame.draw.line(self.screen, dark_color, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)
            pygame.draw.line(self.screen, dark_color, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)

    def _draw_piece(self, shape, offset, color, is_ghost=False):
        off_x, off_y = offset
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(off_x + c, off_y + r, color, is_ghost)
                    
    def _draw_preview_piece(self, piece, pos):
        shape = piece['shape']
        color = self.TETROMINO_COLORS[piece['type']]
        px, py = pos
        cell_size = self.CELL_SIZE * 0.8
        
        shape_w = len(shape[0]) * cell_size
        shape_h = len(shape) * cell_size
        offset_x = px - shape_w / 2
        offset_y = py - shape_h / 2
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(offset_x + c * cell_size, offset_y + r * cell_size, cell_size, cell_size)
                    pygame.draw.rect(self.screen, color, rect)
    
    def _render_line_clear_flash(self):
        for r in self.cleared_lines_indices:
            flash_rect = pygame.Rect(
                self.BOARD_X_OFFSET,
                self.BOARD_Y_OFFSET + r * self.CELL_SIZE,
                self.BOARD_PIXEL_WIDTH,
                self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, (255, 255, 255), flash_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fall_speed": self.fall_speed,
            "game_over": self.game_over,
        }
    
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a way to render the game and get keyboard input.
    # The following is a simple example.
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Tetris-like Puzzle Game")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    running = True
    while running:
        # Default action is no-op
        action = [0, 0, 0] 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Movement (action[0])
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        else: action[0] = 0
            
        # Space (action[1])
        if keys[pygame.K_SPACE]: action[1] = 1
        
        # Shift (action[2])
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()