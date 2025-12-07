
# Generated: 2025-08-27T17:42:36.244660
# Source Brief: brief_01618.md
# Brief Index: 1618

        
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
        "Controls: ←→ to move, ↓ for soft drop, ↑ to rotate. Space for hard drop, Shift to hold piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically rotate and place falling blocks to clear lines and achieve a target score."
    )

    # Frames auto-advance for real-time gameplay.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)
    
    # Piece shapes and colors
    PIECE_SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 0], [0, 1, 1]],  # Z
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1], [1, 1]],  # O
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
    ]
    PIECE_COLORS = [
        (0, 255, 255),  # I (Cyan)
        (255, 0, 0),    # Z (Red)
        (0, 255, 0),    # S (Green)
        (160, 0, 255),  # T (Purple)
        (255, 255, 0),  # O (Yellow)
        (255, 165, 0),  # L (Orange)
        (0, 0, 255),    # J (Blue)
    ]
    
    # Game settings
    WIN_CONDITION_LINES = 20
    MAX_STEPS = 10000
    INITIAL_FALL_DELAY = 30  # Ticks per grid cell drop
    FALL_DELAY_DECREMENT_INTERVAL = 5 # lines
    FALL_DELAY_DECREMENT_AMOUNT = 2 # ticks
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Grid position calculation
        self.grid_render_x = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.grid_render_y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_delay = self.INITIAL_FALL_DELAY
        self.fall_counter = 0
        self.line_clear_animation = None
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call for debugging

    def _create_new_piece(self):
        piece_idx = self.np_random.integers(0, len(self.PIECE_SHAPES))
        shape = self.PIECE_SHAPES[piece_idx]
        return {
            "shape": shape,
            "color_idx": piece_idx,
            "x": self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.current_piece = self._create_new_piece()
        self.next_piece = self._create_new_piece()
        self.held_piece = None
        self.can_hold = True
        
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        
        self.fall_delay = self.INITIAL_FALL_DELAY
        self.fall_counter = 0
        self.line_clear_animation = None
        
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), -100, True, False, self._get_info()
            
        if self.line_clear_animation:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self._finalize_line_clear()
            return self._get_observation(), 0, False, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # 1. Handle player actions
        # Action: Hold Piece (Shift)
        if shift_held and self.can_hold:
            self._hold_piece()
            # SFX: Hold piece sound
        
        # Action: Hard Drop (Space)
        elif space_held:
            reward += self._hard_drop()
            # SFX: Hard drop thud
        
        # Action: Movement/Rotation
        else:
            self._handle_movement(movement)
            # SFX: Move/Rotate clicks
        
        # 2. Game Physics: Automatic fall
        self.fall_counter += 1
        if self.fall_counter >= self.fall_delay:
            self.fall_counter = 0
            moved_down = self._move_piece(0, 1)
            if moved_down:
                reward += 0.1 # Reward for natural fall

        # 3. Check for landing and line clears
        if not self._is_valid_position(self.current_piece, dy=1):
            self._place_piece()
            lines = self._check_and_initiate_clear()
            if lines > 0:
                # Reward for clearing lines
                line_rewards = {1: 1, 2: 3, 3: 6, 4: 10}
                reward += line_rewards.get(lines, 0)
                # Traditional score for display
                score_map = {1: 100, 2: 300, 3: 500, 4: 800}
                self.score += score_map.get(lines, 0) * (self.lines_cleared // 10 + 1)
                self.lines_cleared += lines
                
                # Increase difficulty
                self.fall_delay = max(5, self.INITIAL_FALL_DELAY - (self.lines_cleared // self.FALL_DELAY_DECREMENT_INTERVAL) * self.FALL_DELAY_DECREMENT_AMOUNT)
                # SFX: Line clear fanfare
            else:
                # SFX: Soft landing sound
                pass

            self._spawn_new_piece()

        # 4. Check for termination conditions
        terminated = self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS
        if terminated:
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                reward += 100  # Win bonus
                self.score += 1000
            elif self.game_over:
                reward -= 100  # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        # 0=none, 1=up(rotate), 2=down, 3=left, 4=right
        if movement == 1: # Rotate
            self._rotate_piece()
        elif movement == 2: # Soft Drop
            self.fall_counter += 5 # Accelerate fall
        elif movement == 3: # Left
            self._move_piece(-1, 0)
        elif movement == 4: # Right
            self._move_piece(1, 0)

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece, dx=dx, dy=dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self):
        shape = self.current_piece['shape']
        rotated_shape = list(zip(*shape[::-1]))
        
        original_x = self.current_piece['x']
        test_x = original_x
        
        # Basic wall kick logic
        for offset in [0, 1, -1, 2, -2]:
            self.current_piece['x'] = original_x + offset
            if self._is_valid_position({"shape": rotated_shape, "x": self.current_piece['x'], "y": self.current_piece['y']}):
                self.current_piece['shape'] = rotated_shape
                return
        
        self.current_piece['x'] = original_x # Revert if no valid rotation found

    def _hard_drop(self):
        dy = 0
        while self._is_valid_position(self.current_piece, dy=dy + 1):
            dy += 1
        self.current_piece['y'] += dy
        return dy * 0.1 # Reward for dropping

    def _hold_piece(self):
        self.can_hold = False
        if self.held_piece:
            self.current_piece, self.held_piece = self.held_piece, self.current_piece
            self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 0
        else:
            self.held_piece = self.current_piece
            self._spawn_new_piece(from_hold=True)
            
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _place_piece(self):
        shape = self.current_piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_y, grid_x = self.current_piece['y'] + r, self.current_piece['x'] + c
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y][grid_x] = self.current_piece['color_idx'] + 1

    def _spawn_new_piece(self, from_hold=False):
        self.current_piece = self.next_piece
        if not from_hold:
            self.next_piece = self._create_new_piece()
        self.can_hold = True
        self.fall_counter = 0
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def _check_and_initiate_clear(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if all(self.grid[r]):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            self.line_clear_animation = {
                'lines': lines_to_clear,
                'timer': 10 # Animation duration in frames
            }
        return len(lines_to_clear)

    def _finalize_line_clear(self):
        lines = self.line_clear_animation['lines']
        for r in sorted(lines, reverse=True):
            self.grid = np.delete(self.grid, r, axis=0)
            new_row = np.zeros((1, self.GRID_WIDTH), dtype=int)
            self.grid = np.insert(self.grid, 0, new_row, axis=0)
        self.line_clear_animation = None

    def _is_valid_position(self, piece, dx=0, dy=0):
        shape = piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_y, grid_x = piece['y'] + r + dy, piece['x'] + c + dx
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_y][grid_x] != 0:
                        return False
        return True

    def _get_ghost_y(self):
        dy = 0
        while self._is_valid_position(self.current_piece, dy=dy + 1):
            dy += 1
        return self.current_piece['y'] + dy

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_render_x, self.grid_render_y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        
        # Draw landed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] > 0:
                    color_idx = int(self.grid[r][c]) - 1
                    self._draw_cell(c, r, self.PIECE_COLORS[color_idx])

        # Draw ghost piece
        if not self.game_over and self.current_piece:
            ghost_y = self._get_ghost_y()
            ghost_piece = self.current_piece.copy()
            ghost_piece['y'] = ghost_y
            self._draw_piece(ghost_piece, is_ghost=True)

        # Draw current piece
        if not self.game_over and self.current_piece:
            self._draw_piece(self.current_piece)
            
        # Draw line clear animation
        if self.line_clear_animation:
            flash_color = self.COLOR_WHITE if (self.line_clear_animation['timer'] // 2) % 2 == 0 else self.COLOR_GRID
            for r in self.line_clear_animation['lines']:
                for c in range(self.GRID_WIDTH):
                    self._draw_cell(c, r, flash_color)

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_render_x + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (x, self.grid_render_y), (x, self.grid_render_y + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_render_y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_BG, (self.grid_render_x, y), (self.grid_render_x + self.GRID_WIDTH * self.CELL_SIZE, y))

    def _draw_piece(self, piece, is_ghost=False):
        shape = piece['shape']
        color = self.PIECE_COLORS[piece['color_idx']]
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(piece['x'] + c, piece['y'] + r, color, is_ghost)
    
    def _draw_cell(self, grid_c, grid_r, color, is_ghost=False):
        x = self.grid_render_x + grid_c * self.CELL_SIZE
        y = self.grid_render_y + grid_r * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2) # Just the outline
        else:
            # Main block color
            pygame.draw.rect(self.screen, color, rect)
            # 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, (x, y), (x + self.CELL_SIZE - 1, y))
            pygame.draw.line(self.screen, highlight, (x, y), (x, y + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1))
            pygame.draw.line(self.screen, shadow, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Lines
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 50))
        
        # Next Piece
        self._render_side_box(self.SCREEN_WIDTH - 150, 50, "NEXT", self.next_piece)
        
        # Held Piece
        self._render_side_box(self.SCREEN_WIDTH - 150, 200, "HOLD", self.held_piece)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _render_side_box(self, x, y, title, piece):
        box_rect = pygame.Rect(x, y, 130, 120)
        pygame.draw.rect(self.screen, self.COLOR_GRID, box_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, box_rect, 2, border_radius=5)
        
        title_text = self.font_small.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_text, (x + 10, y + 5))
        
        if piece:
            shape = piece['shape']
            color = self.PIECE_COLORS[piece['color_idx']]
            
            start_x = x + (130 - len(shape[0]) * self.CELL_SIZE) / 2
            start_y = y + (120 - len(shape) * self.CELL_SIZE) / 2 + 10
            
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        cell_x = start_x + c * self.CELL_SIZE
                        cell_y = start_y + r * self.CELL_SIZE
                        rect = pygame.Rect(cell_x, cell_y, self.CELL_SIZE, self.CELL_SIZE)
                        pygame.draw.rect(self.screen, color, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a way to render the game and capture keys.
    # This is outside the standard Gym loop but useful for testing.
    
    # Set render_mode to "human" if you add a human render mode
    # For now, we'll use a pygame window to show the rgb_array
    
    pygame.display.set_caption("Puzzle Block Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Action state
    movement = 0 # 0=none, 1=up(rotate), 2=down, 3=left, 4=right
    space_held = 0
    shift_held = 0
    
    while not terminated:
        # Action mapping:
        # actions[0]: Movement (0=none, 1=up(rotate), 2=down, 3=left, 4=right)
        # actions[1]: Space button (0=released, 1=held)
        # actions[2]: Shift button (0=released, 1=held)
        
        # Reset actions for this frame
        movement = 0
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for manual play
        
    env.close()
    print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines_cleared']}")