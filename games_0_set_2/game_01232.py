
# Generated: 2025-08-27T16:27:46.895814
# Source Brief: brief_01232.md
# Brief Index: 1232

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop, space for hard drop. Shift to rotate. Clear 10 lines to win."
    )

    # Short, user-facing description of the game
    game_description = (
        "Clear lines of colorful blocks in a fast-paced, grid-based puzzle game. Stack pieces, complete rows, and watch them disappear."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    
    # Visuals
    CELL_SIZE = 18
    BOARD_X_OFFSET = (SCREEN_WIDTH - BOARD_WIDTH * CELL_SIZE) // 2 - 100
    BOARD_Y_OFFSET = (SCREEN_HEIGHT - BOARD_HEIGHT * CELL_SIZE) // 2
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_FLASH = (255, 255, 255)

    PIECE_SHAPES = {
        'I': [[1, 1, 1, 1]],
        'O': [[1, 1], [1, 1]],
        'T': [[0, 1, 0], [1, 1, 1]],
        'S': [[0, 1, 1], [1, 1, 0]],
        'Z': [[1, 1, 0], [0, 1, 1]],
        'J': [[1, 0, 0], [1, 1, 1]],
        'L': [[0, 0, 1], [1, 1, 1]],
    }
    
    PIECE_COLORS = {
        'I': (0, 240, 240),   # Cyan
        'O': (240, 240, 0),   # Yellow
        'T': (160, 0, 240),   # Purple
        'S': (0, 240, 0),     # Green
        'Z': (240, 0, 0),     # Red
        'J': (0, 0, 240),     # Blue
        'L': (240, 160, 0),   # Orange
    }

    # Gameplay
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 10000
    INITIAL_FALL_SPEED_SECONDS = 1.0
    FALL_SPEED_DECREMENT = 0.05
    SOFT_DROP_MULTIPLIER = 10.0
    DAS_DELAY = 8 # frames
    DAS_RATE = 2 # frames

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(pygame.font.get_default_font(), 36)
        self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.font_small = pygame.font.Font(pygame.font.get_default_font(), 16)

        # Initialize state variables
        self.board = None
        self.current_piece = None
        self.next_pieces = None
        self.score = None
        self.lines_cleared = None
        self.steps = None
        self.game_over = None
        self.fall_timer = None
        self.fall_speed = None
        
        # Action handling state
        self.last_space_held = False
        self.last_shift_held = False
        self.move_key_held_frames = 0
        self.last_move_action = 0

        # Animation state
        self.lines_to_clear_anim = []
        self.line_clear_anim_timer = 0
        self.LINE_CLEAR_ANIM_DURATION = 6 # frames

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=int)
        
        self.piece_bag = list(self.PIECE_SHAPES.keys())
        random.shuffle(self.piece_bag)
        self.next_pieces = deque(self.piece_bag[:4])
        
        self._spawn_piece()
        
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_speed = self.INITIAL_FALL_SPEED_SECONDS
        self.fall_timer = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        self.move_key_held_frames = 0
        self.last_move_action = 0

        self.lines_to_clear_anim = []
        self.line_clear_anim_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle animations first ---
        if self.line_clear_anim_timer > 0:
            self.line_clear_anim_timer -= 1
            if self.line_clear_anim_timer == 0:
                self._execute_line_clear()
            # No other game logic happens during the clear animation
            return self._get_observation(), 0, False, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for time passing

        # --- Unpack and process actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        
        # Handle DAS (Delayed Auto-Shift) for horizontal movement
        if movement in [3, 4]: # left or right
            if movement != self.last_move_action:
                self.move_key_held_frames = 0
                
            if self.move_key_held_frames == 0:
                self._move_piece(movement - 3 if movement == 3 else 1)
            elif self.move_key_held_frames > self.DAS_DELAY and (self.steps % self.DAS_RATE == 0):
                self._move_piece(movement - 3 if movement == 3 else 1)
            
            self.move_key_held_frames += 1

            # "Safe action" penalty
            if not self._check_collision(self.current_piece, (0, 1)):
                reward -= 0.2
        else:
            self.move_key_held_frames = 0

        self.last_move_action = movement

        if shift_pressed:
            self._rotate_piece()

        if space_pressed:
            # Hard drop
            drop_distance = 0
            while not self._check_collision(self.current_piece, (0, drop_distance + 1)):
                drop_distance += 1
            self.current_piece['y'] += drop_distance
            self._lock_piece()
        else:
            # --- Game Tick Logic ---
            is_soft_dropping = (movement == 2)
            time_to_advance = 1 / 30.0 # Assuming 30fps
            if is_soft_dropping:
                time_to_advance *= self.SOFT_DROP_MULTIPLIER

            self.fall_timer += time_to_advance
            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                self._move_piece(0, 1) # Move down

        # --- Update rewards and check termination ---
        reward += self._calculate_reward()
        terminated = self._check_termination()
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_piece(self):
        shape_key = self.next_pieces.popleft()
        
        if len(self.next_pieces) < 4:
            new_bag = list(self.PIECE_SHAPES.keys())
            random.shuffle(new_bag)
            self.next_pieces.extend(new_bag)
            
        self.current_piece = {
            'shape_key': shape_key,
            'shape': self.PIECE_SHAPES[shape_key],
            'color_idx': list(self.PIECE_SHAPES.keys()).index(shape_key) + 1,
            'x': self.BOARD_WIDTH // 2 - len(self.PIECE_SHAPES[shape_key][0]) // 2,
            'y': 0,
        }
        if self._check_collision(self.current_piece):
            self.game_over = True

    def _check_collision(self, piece, offset=(0, 0)):
        px, py = piece['x'] + offset[0], piece['y'] + offset[1]
        for r_idx, row in enumerate(piece['shape']):
            for c_idx, cell in enumerate(row):
                if cell:
                    board_x, board_y = px + c_idx, py + r_idx
                    if not (0 <= board_x < self.BOARD_WIDTH and 0 <= board_y < self.BOARD_HEIGHT):
                        return True  # Wall collision
                    if self.board[board_y, board_x] != 0:
                        return True  # Block collision
        return False

    def _move_piece(self, dx, dy=0):
        if not self._check_collision(self.current_piece, (dx, dy)):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
        elif dy > 0: # If moving down results in collision
            self._lock_piece()

    def _rotate_piece(self):
        # sound: piece_rotate.wav
        original_shape = self.current_piece['shape']
        rotated_shape = list(zip(*original_shape[::-1]))
        
        # Simple wall kick test (just left and right)
        for kick_x in [0, -1, 1, -2, 2]:
            temp_piece = self.current_piece.copy()
            temp_piece['shape'] = rotated_shape
            if not self._check_collision(temp_piece, (kick_x, 0)):
                self.current_piece['shape'] = rotated_shape
                self.current_piece['x'] += kick_x
                return
    
    def _lock_piece(self):
        # sound: piece_lock.wav
        px, py = self.current_piece['x'], self.current_piece['y']
        for r_idx, row in enumerate(self.current_piece['shape']):
            for c_idx, cell in enumerate(row):
                if cell:
                    self.board[py + r_idx, px + c_idx] = self.current_piece['color_idx']
        
        self._check_and_clear_lines()
        self._spawn_piece()
    
    def _check_and_clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.board) if np.all(row != 0)]
        if lines_to_clear:
            self.lines_to_clear_anim = lines_to_clear
            self.line_clear_anim_timer = self.LINE_CLEAR_ANIM_DURATION

    def _execute_line_clear(self):
        num_cleared = len(self.lines_to_clear_anim)
        if num_cleared == 0: return

        # sound: line_clear.wav for 1-3, tetris_clear.wav for 4
        self.lines_cleared += num_cleared
        
        # Update difficulty
        if self.lines_cleared % 2 == 0 and self.lines_cleared > 0:
            self.fall_speed = max(0.1, self.INITIAL_FALL_SPEED_SECONDS - (self.lines_cleared // 2) * self.FALL_SPEED_DECREMENT)

        # Rewards for line clears
        if num_cleared == 1: self.score += 1
        elif num_cleared == 2: self.score += 2
        elif num_cleared == 3: self.score += 3
        elif num_cleared == 4: self.score += 5

        # Remove cleared lines
        self.board = np.delete(self.board, self.lines_to_clear_anim, axis=0)
        # Add new empty lines at the top
        new_lines = np.zeros((num_cleared, self.BOARD_WIDTH), dtype=int)
        self.board = np.vstack((new_lines, self.board))

        self.lines_to_clear_anim = []

    def _calculate_reward(self):
        # Rewards are handled at the event points (line clear, game over)
        # This function is a placeholder for any continuous rewards if needed
        return 0

    def _check_termination(self):
        if self.game_over:
            self.score -= 100 # Loss penalty
            return True
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.score += 100 # Win bonus
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

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
        # Draw grid
        for i in range(self.BOARD_WIDTH + 1):
            x = self.BOARD_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.BOARD_Y_OFFSET), (x, self.BOARD_Y_OFFSET + self.BOARD_HEIGHT * self.CELL_SIZE))
        for i in range(self.BOARD_HEIGHT + 1):
            y = self.BOARD_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.BOARD_X_OFFSET, y), (self.BOARD_X_OFFSET + self.BOARD_WIDTH * self.CELL_SIZE, y))

        # Draw locked blocks
        for r in range(self.BOARD_HEIGHT):
            for c in range(self.BOARD_WIDTH):
                if self.board[r, c] != 0:
                    color_key = list(self.PIECE_SHAPES.keys())[int(self.board[r, c]) - 1]
                    self._draw_cell(c, r, self.PIECE_COLORS[color_key])

        # Draw line clear animation
        if self.line_clear_anim_timer > 0:
            for r in self.lines_to_clear_anim:
                flash_rect = pygame.Rect(self.BOARD_X_OFFSET, self.BOARD_Y_OFFSET + r * self.CELL_SIZE, self.BOARD_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_FLASH, flash_rect)

        # Draw current piece and ghost
        if self.current_piece and not self.game_over:
            # Ghost piece
            ghost_y_offset = 0
            while not self._check_collision(self.current_piece, (0, ghost_y_offset + 1)):
                ghost_y_offset += 1
            
            px_ghost, py_ghost = self.current_piece['x'], self.current_piece['y'] + ghost_y_offset
            color_key = self.current_piece['shape_key']
            ghost_color = self.PIECE_COLORS[color_key]
            for r_idx, row in enumerate(self.current_piece['shape']):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_cell(px_ghost + c_idx, py_ghost + r_idx, ghost_color, is_ghost=True)

            # Active piece
            px, py = self.current_piece['x'], self.current_piece['y']
            for r_idx, row in enumerate(self.current_piece['shape']):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_cell(px + c_idx, py + r_idx, self.PIECE_COLORS[color_key])

    def _draw_cell(self, c, r, color, is_ghost=False):
        rect = pygame.Rect(
            self.BOARD_X_OFFSET + c * self.CELL_SIZE,
            self.BOARD_Y_OFFSET + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2) # Draw outline
        else:
            # Main block color
            pygame.draw.rect(self.screen, color, rect)
            # 3D effect
            highlight_color = tuple(min(255, val + 50) for val in color)
            shadow_color = tuple(max(0, val - 50) for val in color)
            pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright)
            pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft)
            pygame.draw.line(self.screen, shadow_color, rect.bottomleft, rect.bottomright)
            pygame.draw.line(self.screen, shadow_color, rect.topright, rect.bottomright)

    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render("SCORE", True, self.COLOR_UI_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (30, 30))
        self.screen.blit(score_val, (30, 60))

        # Lines display
        lines_text = self.font_medium.render("LINES", True, self.COLOR_UI_TEXT)
        lines_val = self.font_large.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (30, 120))
        self.screen.blit(lines_val, (30, 150))

        # Next piece display
        next_text = self.font_medium.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 150, 30))
        
        preview_box_x = self.SCREEN_WIDTH - 160
        preview_box_y = 60
        
        if self.next_pieces:
            next_shape_key = self.next_pieces[0]
            next_shape = self.PIECE_SHAPES[next_shape_key]
            next_color = self.PIECE_COLORS[next_shape_key]
            
            shape_w = len(next_shape[0])
            shape_h = len(next_shape)
            
            for r_idx, row in enumerate(next_shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        x = preview_box_x + c_idx * self.CELL_SIZE + (4-shape_w)*self.CELL_SIZE/2
                        y = preview_box_y + r_idx * self.CELL_SIZE + (4-shape_h)*self.CELL_SIZE/2
                        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                        self._draw_cell(x / self.CELL_SIZE, y / self.CELL_SIZE, next_color) # Use draw_cell for consistent look

        # Game Over or Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                msg_text = "YOU WIN!"
            else:
                msg_text = "GAME OVER"
                
            msg_render = self.font_large.render(msg_text, True, self.COLOR_FLASH)
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_render, msg_rect)

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game with keyboard controls
    # To use, you need to make Pygame render to a window instead of a surface.
    # 1. In __init__, change `self.screen = pygame.Surface(...)` to `self.screen = pygame.display.set_mode(...)`
    # 2. In _get_observation, add `pygame.display.flip()`
    # 3. Uncomment and run this block.

    # Example of how to run the environment
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Simple random agent loop
    for _ in range(1000):
        if done:
            break
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        # To see the output, you would need to save the `obs` as an image
        # from PIL import Image
        # img = Image.fromarray(obs)
        # img.save(f"frame_{env.steps:04d}.png")

    print(f"Random agent finished with score: {info['score']} and total reward: {total_reward:.2f}")
    env.close()