
# Generated: 2025-08-28T06:24:04.259530
# Source Brief: brief_05882.md
# Brief Index: 5882

        
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
        "Controls: ←→ to move, ↑/Space to rotate. ↓ for soft drop, Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Classic falling block puzzle game. Clear lines to score points and prevent the stack from reaching the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.BOARD_WIDTH, self.BOARD_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.BOARD_X_OFFSET = (self.WIDTH - self.BOARD_WIDTH * self.BLOCK_SIZE) // 2
        self.BOARD_Y_OFFSET = (self.HEIGHT - self.BOARD_HEIGHT * self.BLOCK_SIZE) // 2
        
        self.FALL_SPEED = 20 # Ticks per row drop
        self.WIN_CONDITION_LINES = 25
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_PANEL = (30, 30, 45)
        self.TETROMINO_COLORS = [
            (30, 30, 45),    # 0: Empty/Panel color
            (0, 240, 240),   # 1: I - Cyan
            (240, 240, 0),   # 2: O - Yellow
            (160, 0, 240),   # 3: T - Purple
            (0, 0, 240),     # 4: J - Blue
            (240, 160, 0),   # 5: L - Orange
            (0, 240, 0),     # 6: S - Green
            (240, 0, 0),     # 7: Z - Red
        ]

        # --- Tetromino Shapes ---
        # Coordinates are relative to a pivot point
        self.TETROMINOES = {
            'I': [[[-1, 0], [0, 0], [1, 0], [2, 0]], [[0, -1], [0, 0], [0, 1], [0, 2]]],
            'O': [[[0, 0], [1, 0], [0, 1], [1, 1]]],
            'T': [[[-1, 0], [0, 0], [1, 0], [0, -1]], [[0, -1], [0, 0], [0, 1], [-1, 0]], [[-1, 0], [0, 0], [1, 0], [0, 1]], [[0, -1], [0, 0], [0, 1], [1, 0]]],
            'J': [[[-1, -1], [-1, 0], [0, 0], [1, 0]], [[0, -1], [0, 0], [0, 1], [1, -1]], [[-1, 0], [0, 0], [1, 0], [1, 1]], [[0, -1], [0, 0], [0, 1], [-1, 1]]],
            'L': [[[1, -1], [-1, 0], [0, 0], [1, 0]], [[0, -1], [0, 0], [0, 1], [-1, -1]], [[-1, 0], [0, 0], [1, 0], [-1, 1]], [[0, -1], [0, 0], [0, 1], [1, 1]]],
            'S': [[[-1, 0], [0, 0], [0, -1], [1, -1]], [[0, -1], [0, 0], [1, 0], [1, 1]]],
            'Z': [[[0, -1], [1, -1], [-1, 0], [0, 0]], [[0, 0], [1, 0], [0, 1], [1, -1]]]
        }
        self.TETROMINO_NAMES = list(self.TETROMINOES.keys())
        self.TETROMINO_MAP = {name: i + 1 for i, name in enumerate(self.TETROMINO_NAMES)}

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Game State ---
        self.board = None
        self.current_piece = None
        self.next_pieces = []
        self.lines_cleared = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_counter = 0
        self.line_clear_animation = None
        self.last_action_reward = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=int)
        self.lines_cleared = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_counter = 0
        self.line_clear_animation = None
        self.last_action_reward = 0
        
        self.next_pieces = [self.np_random.choice(self.TETROMINO_NAMES) for _ in range(3)]
        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Action Handling ---
        piece_landed = False
        
        # 1. Hard Drop (takes precedence and ends the turn)
        if shift_held:
            # sfx: hard_drop_sound
            moved_down = 0
            while self._is_valid_position(offset_y=1):
                self.current_piece['pos'][0] += 1
                moved_down += 1
            reward += moved_down * 0.02 # Small reward for distance dropped
            piece_landed = True
        else:
            # 2. Rotation
            if movement == 1: # Up -> Rotate CW
                self._rotate_piece(1)
            if space_held: # Space -> Rotate CCW
                self._rotate_piece(-1)
            
            # 3. Horizontal Movement
            if movement == 3: # Left
                self._move_piece(-1)
            elif movement == 4: # Right
                self._move_piece(1)
            
            # 4. Gravity / Soft Drop
            is_soft_drop = (movement == 2)
            self.fall_counter += 5 if is_soft_drop else 1
            
            if self.fall_counter >= self.FALL_SPEED:
                self.fall_counter = 0
                if self._is_valid_position(offset_y=1):
                    self.current_piece['pos'][0] += 1
                    if is_soft_drop:
                        reward += 0.01 # Small reward for soft dropping
                else:
                    piece_landed = True

        # --- Piece Landing Logic ---
        if piece_landed:
            # sfx: piece_land_sound
            landing_reward = self._place_piece_and_spawn_new()
            reward += landing_reward
        
        # --- Termination Conditions ---
        terminated = self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.game_over:
                reward = -100
            elif self.lines_cleared >= self.WIN_CONDITION_LINES:
                reward += 100
            
        self.score = self.lines_cleared # Use lines cleared as score for simplicity
        self.last_action_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_board()
        self._draw_pieces()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }

    # --- Game Logic Helpers ---
    def _spawn_new_piece(self):
        self.current_piece = {
            'shape_name': self.next_pieces.pop(0),
            'rotation': 0,
            'pos': [0, self.BOARD_WIDTH // 2 - 1]
        }
        self.current_piece['color_index'] = self.TETROMINO_MAP[self.current_piece['shape_name']]
        self.next_pieces.append(self.np_random.choice(self.TETROMINO_NAMES))
        
        if not self._is_valid_position():
            self.game_over = True
            # sfx: game_over_sound

    def _place_piece_and_spawn_new(self):
        reward = 0
        
        height_before = self._get_board_height()
        
        shape_coords = self._get_current_piece_coords()
        for r, c in shape_coords:
            if 0 <= r < self.BOARD_HEIGHT and 0 <= c < self.BOARD_WIDTH:
                self.board[r, c] = self.current_piece['color_index']
        
        height_after = self._get_board_height()
        if height_after > height_before:
            reward -= 0.1 * (height_after - height_before)

        lines_cleared, cleared_rows = self._clear_lines()
        if lines_cleared > 0:
            # sfx: line_clear_sound
            self.lines_cleared += lines_cleared
            reward_map = {1: 1, 2: 3, 3: 7, 4: 15}
            reward += reward_map.get(lines_cleared, 0)
            self.line_clear_animation = {'rows': cleared_rows, 'timer': 10}

        self._spawn_new_piece()
        return reward

    def _clear_lines(self):
        full_rows = [r for r in range(self.BOARD_HEIGHT) if np.all(self.board[r] > 0)]
        
        if not full_rows:
            return 0, []

        self.board = np.delete(self.board, full_rows, axis=0)
        new_rows = np.zeros((len(full_rows), self.BOARD_WIDTH), dtype=int)
        self.board = np.vstack((new_rows, self.board))

        return len(full_rows), full_rows

    def _get_board_height(self):
        if not self.board.any():
            return 0
        non_empty_rows = np.where(self.board.any(axis=1))[0]
        if len(non_empty_rows) == 0:
            return 0
        return self.BOARD_HEIGHT - non_empty_rows.min()

    def _get_current_piece_coords(self, piece=None):
        if piece is None:
            piece = self.current_piece
        
        shape = self.TETROMINOES[piece['shape_name']][piece['rotation']]
        return [[r + piece['pos'][0], c + piece['pos'][1]] for r, c in shape]

    def _is_valid_position(self, offset_x=0, offset_y=0, piece=None):
        if piece is None:
            piece = self.current_piece

        temp_piece = piece.copy()
        temp_piece['pos'] = [piece['pos'][0] + offset_y, piece['pos'][1] + offset_x]
        
        coords = self._get_current_piece_coords(temp_piece)
        for r, c in coords:
            if not (0 <= c < self.BOARD_WIDTH and r < self.BOARD_HEIGHT):
                return False
            if r >= 0 and self.board[r, c] > 0:
                return False
        return True

    def _move_piece(self, dx):
        if self._is_valid_position(offset_x=dx):
            self.current_piece['pos'][1] += dx
            # sfx: move_sound

    def _rotate_piece(self, direction):
        # sfx: rotate_sound
        num_rotations = len(self.TETROMINOES[self.current_piece['shape_name']])
        original_rotation = self.current_piece['rotation']
        
        self.current_piece['rotation'] = (self.current_piece['rotation'] + direction) % num_rotations
        
        # Simplified Wall Kick
        if not self._is_valid_position():
            if self._is_valid_position(offset_x=1):
                self.current_piece['pos'][1] += 1
            elif self._is_valid_position(offset_x=-1):
                self.current_piece['pos'][1] -= 1
            else: # Revert rotation
                self.current_piece['rotation'] = original_rotation

    # --- Rendering Helpers ---
    def _draw_board(self):
        board_rect = pygame.Rect(self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET, self.BOARD_WIDTH * self.BLOCK_SIZE, self.BOARD_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, board_rect)
        for x in range(self.BOARD_WIDTH + 1):
            px = self.BOARD_X_OFFSET + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.BOARD_Y_OFFSET), (px, self.BOARD_Y_OFFSET + self.BOARD_HEIGHT * self.BLOCK_SIZE))
        for y in range(self.BOARD_HEIGHT + 1):
            py = self.BOARD_Y_OFFSET + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, py), (self.BOARD_X_OFFSET + self.BOARD_WIDTH * self.BLOCK_SIZE, py))
        
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                if self.board[r, c] > 0:
                    self._draw_block(c, r, self.board[r, c])

    def _draw_pieces(self):
        ghost_piece = self.current_piece.copy()
        while self._is_valid_position(piece=ghost_piece, offset_y=1):
            ghost_piece['pos'][0] += 1
        
        ghost_coords = self._get_current_piece_coords(ghost_piece)
        for r, c in ghost_coords:
            if r >= 0:
                self._draw_block(c, r, self.current_piece['color_index'], is_ghost=True)

        current_coords = self._get_current_piece_coords()
        for r, c in current_coords:
            if r >= 0:
                self._draw_block(c, r, self.current_piece['color_index'])
        
        if self.line_clear_animation:
            timer = self.line_clear_animation['timer']
            alpha = int(255 * (timer / 10.0))
            color = (255, 255, 255, alpha)
            
            for r in self.line_clear_animation['rows']:
                rect = pygame.Rect(self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET + r * self.BLOCK_SIZE, self.BOARD_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
                flash_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                flash_surface.fill(color)
                self.screen.blit(flash_surface, rect.topleft)

            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self.line_clear_animation = None

    def _draw_block(self, c, r, color_index, is_ghost=False):
        px = self.BOARD_X_OFFSET + c * self.BLOCK_SIZE
        py = self.BOARD_Y_OFFSET + r * self.BLOCK_SIZE
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        color = self.TETROMINO_COLORS[color_index]
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
        else:
            light_color = tuple(min(255, val + 50) for val in color)
            dark_color = tuple(max(0, val - 50) for val in color)
            
            pygame.draw.rect(self.screen, dark_color, rect)
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, inner_rect)
            
            pygame.gfxdraw.filled_polygon(self.screen, [(px+1, py+1), (px + self.BLOCK_SIZE-1, py+1), (px + self.BLOCK_SIZE-2, py+2), (px+2, py+2)], light_color)
            pygame.gfxdraw.filled_polygon(self.screen, [(px+1, py+1), (px+1, py + self.BLOCK_SIZE-1), (px+2, py + self.BLOCK_SIZE-2), (px+2, py+2)], light_color)

    def _draw_ui(self):
        next_panel_rect = pygame.Rect(self.BOARD_X_OFFSET + self.BOARD_WIDTH * self.BLOCK_SIZE + 10, self.BOARD_Y_OFFSET, 100, 160)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, next_panel_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_panel_rect, 2)
        
        text_surf = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (next_panel_rect.centerx - text_surf.get_width() // 2, next_panel_rect.top + 10))
        
        for i, shape_name in enumerate(self.next_pieces[:3]):
            shape = self.TETROMINOES[shape_name][0]
            color_idx = self.TETROMINO_MAP[shape_name]
            for r_off, c_off in shape:
                px = int(next_panel_rect.centerx + c_off * self.BLOCK_SIZE // 1.5)
                py = int(next_panel_rect.top + 55 + i * 45 + r_off * self.BLOCK_SIZE // 1.5)
                rect = pygame.Rect(px, py, self.BLOCK_SIZE // 1.5, self.BLOCK_SIZE // 1.5)
                pygame.draw.rect(self.screen, self.TETROMINO_COLORS[color_idx], rect)

        score_panel_rect = pygame.Rect(next_panel_rect.left, next_panel_rect.bottom + 10, 100, 100)
        pygame.draw.rect(self.screen, self.COLOR_PANEL, score_panel_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID, score_panel_rect, 2)
        
        lines_text = self.font_main.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (score_panel_rect.centerx - lines_text.get_width() // 2, score_panel_rect.top + 10))
        
        lines_val = self.font_main.render(f"{self.lines_cleared}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (score_panel_rect.centerx - lines_val.get_width() // 2, score_panel_rect.top + 50))
        
        reward_color = (0, 255, 0) if self.last_action_reward > 0 else ((255, 0, 0) if self.last_action_reward < 0 else self.COLOR_TEXT)
        reward_text = self.font_small.render(f"Reward: {self.last_action_reward:.2f}", True, reward_color)
        self.screen.blit(reward_text, (10, 10))

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Gymnasium Game - Manual Test")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    total_reward = 0
    
    while not done:
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            # Handle keydown for single press actions
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1 # Rotate CW
                elif event.key == pygame.K_SPACE:
                    action[1] = 1 # Rotate CCW
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1 # Hard drop

        # Handle key held for continuous actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_DOWN]:
            action[0] = 2 # Soft drop
        elif keys[pygame.K_LEFT]:
            action[0] = 3 # Move left
        elif keys[pygame.K_RIGHT]:
            action[0] = 4 # Move right

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        pygame.time.wait(30)

    print(f"Game Over! Final Score (Lines): {info['lines_cleared']}, Total Reward: {total_reward:.2f}")
    pygame.quit()