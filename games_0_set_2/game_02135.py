
# Generated: 2025-08-28T03:49:03.023469
# Source Brief: brief_02135.md
# Brief Index: 2135

        
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
        "Controls: ↑ to rotate clockwise, ←→ to move. ↓ for soft drop. Space for hard drop. Shift for counter-clockwise rotation."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based block-dropping puzzle game. Clear lines by filling them completely. Game ends if blocks stack to the top or 10 lines are cleared."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    CELL_SIZE = 18
    
    # Game area position
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 215, 0)
    COLOR_FLASH = (255, 255, 255)
    
    # Tetromino shapes and colors
    PIECE_DATA = {
        'I': {'shape': [(0, -1), (0, 0), (0, 1), (0, 2)], 'color': (0, 240, 240)},
        'O': {'shape': [(0, 0), (0, 1), (1, 0), (1, 1)], 'color': (240, 240, 0)},
        'T': {'shape': [(-1, 0), (0, 0), (1, 0), (0, -1)], 'color': (160, 0, 240)},
        'J': {'shape': [(-1, -1), (-1, 0), (0, 0), (1, 0)], 'color': (0, 0, 240)},
        'L': {'shape': [(1, -1), (-1, 0), (0, 0), (1, 0)], 'color': (240, 160, 0)},
        'S': {'shape': [(-1, 0), (0, 0), (0, -1), (1, -1)], 'color': (0, 240, 0)},
        'Z': {'shape': [(-1, -1), (0, -1), (0, 0), (1, 0)], 'color': (240, 0, 0)}
    }
    PIECE_TYPES = list(PIECE_DATA.keys())
    
    # --- Initialization ---
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
        self.font_main = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 36)
        
        # Initialize state variables (will be properly set in reset)
        self.grid = None
        self.current_piece = None
        self.next_piece_type = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.line_flash_timer = 0
        self.flashing_lines = []
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()

    # --- Gymnasium API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.line_flash_timer = 0
        self.flashing_lines = []
        
        self.next_piece_type = self.np_random.choice(self.PIECE_TYPES)
        self._spawn_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle line clear animation state
        if self.line_flash_timer > 0:
            self.line_flash_timer -= 1
            if self.line_flash_timer == 0:
                self._perform_line_clear()
            # Return a neutral state during the flash
            return self._get_observation(), 0, False, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Action Handling ---
        # Highest priority: Hard Drop
        if space_held:
            # sfx: hard_drop_sound
            down_dist = 0
            while not self._check_collision(self.current_piece['pos_x'], self.current_piece['pos_y'] + 1, self.current_piece['shape']):
                self.current_piece['pos_y'] += 1
                down_dist += 1
            reward += down_dist * 0.01 # Small reward for using hard drop
            self._lock_piece()
        
        else:
            # Rotations
            if movement == 1:  # Rotate Clockwise
                self._rotate_piece(1)
                reward -= 0.02
            if shift_held:  # Rotate Counter-Clockwise
                self._rotate_piece(-1)
                reward -= 0.02
            
            # Horizontal Movement
            if movement == 3:  # Left
                if not self._check_collision(self.current_piece['pos_x'] - 1, self.current_piece['pos_y'], self.current_piece['shape']):
                    self.current_piece['pos_x'] -= 1
                    # sfx: move_sound
                reward -= 0.02
            elif movement == 4:  # Right
                if not self._check_collision(self.current_piece['pos_x'] + 1, self.current_piece['pos_y'], self.current_piece['shape']):
                    self.current_piece['pos_x'] += 1
                    # sfx: move_sound
                reward -= 0.02

            # Gravity and Soft Drop
            soft_drop = (movement == 2)
            if soft_drop:
                reward += 0.1

            # Gravity always moves down one step
            if not self._check_collision(self.current_piece['pos_x'], self.current_piece['pos_y'] + 1, self.current_piece['shape']):
                self.current_piece['pos_y'] += 1
            else:
                self._lock_piece()

        # --- Post-Action State Update ---
        lines = self._find_full_lines()
        if lines:
            self.flashing_lines = lines
            self.line_flash_timer = 4 # frames to flash
            
            # Grant reward immediately for clearing lines
            line_rewards = {1: 1, 2: 3, 3: 7, 4: 15}
            reward += line_rewards.get(len(lines), 0)
            self.score += line_rewards.get(len(lines), 0) * 100
            self.lines_cleared += len(lines)
            # sfx: line_clear_sound

        terminated = self.game_over or self.lines_cleared >= 10 or self.steps >= 1000
        if terminated:
            if self.lines_cleared >= 10:
                reward += 100  # Win bonus
            elif self.game_over:
                reward -= 100  # Loss penalty
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] > 0:
                    color_index = int(self.grid[y, x]) - 1
                    color = self.PIECE_DATA[self.PIECE_TYPES[color_index]]['color']
                    self._draw_block(x, y, color)

        # Draw flashing lines
        if self.line_flash_timer > 0:
            for y in self.flashing_lines:
                pygame.draw.rect(self.screen, self.COLOR_FLASH, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE))

        # Draw ghost piece
        if not self.game_over:
            ghost_y = self.current_piece['pos_y']
            while not self._check_collision(self.current_piece['pos_x'], ghost_y + 1, self.current_piece['shape']):
                ghost_y += 1
            
            for dx, dy in self.current_piece['shape']:
                x, y = self.current_piece['pos_x'] + dx, ghost_y + dy
                if 0 <= y < self.GRID_HEIGHT:
                    rect = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.current_piece['color'], rect, 2)

        # Draw current piece
        if not self.game_over:
            for dx, dy in self.current_piece['shape']:
                self._draw_block(self.current_piece['pos_x'] + dx, self.current_piece['pos_y'] + dy, self.current_piece['color'])

        # Draw grid border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE), 1)

    def _render_ui(self):
        # Score
        score_text = self.font_title.render("SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_SCORE)
        self.screen.blit(score_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y))
        self.screen.blit(score_val, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 30))

        # Lines
        lines_text = self.font_title.render("LINES", True, self.COLOR_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared:02d} / 10", True, self.COLOR_SCORE)
        self.screen.blit(lines_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 80))
        self.screen.blit(lines_val, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 110))

        # Next Piece
        next_text = self.font_title.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y + 160))
        
        next_piece_data = self.PIECE_DATA[self.next_piece_type]
        for dx, dy in next_piece_data['shape']:
            x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 60 + dx * self.CELL_SIZE
            y = self.GRID_Y + 210 + dy * self.CELL_SIZE
            self._draw_block_at_pixel(x, y, next_piece_data['color'])

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, 100), pygame.SRCALPHA)
            overlay.fill((20, 20, 30, 200))
            self.screen.blit(overlay, (self.GRID_X, self.GRID_Y + 100))
            
            end_text = "GAME OVER" if self.lines_cleared < 10 else "YOU WIN!"
            text_surf = self.font_title.render(end_text, True, self.COLOR_SCORE)
            text_rect = text_surf.get_rect(center=(self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE / 2, self.GRID_Y + 150))
            self.screen.blit(text_surf, text_rect)

    def _draw_block(self, grid_x, grid_y, color):
        x, y = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        self._draw_block_at_pixel(x, y, color)

    def _draw_block_at_pixel(self, x, y, color):
        light_color = tuple(min(255, c + 50) for c in color)
        dark_color = tuple(max(0, c - 50) for c in color)
        
        main_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, color, main_rect)
        
        # 3D effect
        pygame.draw.line(self.screen, light_color, (x, y), (x + self.CELL_SIZE - 1, y), 2)
        pygame.draw.line(self.screen, light_color, (x, y), (x, y + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, dark_color, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, dark_color, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)

    # --- Game Logic ---
    def _spawn_piece(self):
        piece_type = self.next_piece_type
        self.next_piece_type = self.np_random.choice(self.PIECE_TYPES)
        
        self.current_piece = {
            'type': piece_type,
            'shape': list(self.PIECE_DATA[piece_type]['shape']),
            'color': self.PIECE_DATA[piece_type]['color'],
            'color_idx': self.PIECE_TYPES.index(piece_type) + 1,
            'pos_x': self.GRID_WIDTH // 2 -1,
            'pos_y': 0,
        }
        
        # Adjust for 'O' and 'I' piece spawn centering
        if piece_type == 'O':
            self.current_piece['pos_y'] = -1
        if piece_type == 'I':
            self.current_piece['pos_x'] = self.GRID_WIDTH // 2 -1
            self.current_piece['pos_y'] = -1

        if self._check_collision(self.current_piece['pos_x'], self.current_piece['pos_y'], self.current_piece['shape']):
            self.game_over = True
            # sfx: game_over_sound

    def _check_collision(self, piece_x, piece_y, shape):
        for dx, dy in shape:
            x, y = piece_x + dx, piece_y + dy
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True
            if y >= 0 and self.grid[y, x] > 0:
                return True
        return False

    def _rotate_piece(self, direction):
        # sfx: rotate_sound
        if self.current_piece['type'] == 'O':
            return # 'O' piece doesn't rotate

        original_shape = list(self.current_piece['shape'])
        rotated_shape = []
        for dx, dy in original_shape:
            # Rotation matrix: [cos, -sin], [sin, cos]
            # 90 deg clockwise (dir=1): [0, 1], [-1, 0] -> (dy, -dx)
            # 90 deg c-clockwise (dir=-1): [0, -1], [1, 0] -> (-dy, dx)
            rotated_shape.append((dy * -direction, dx * direction))
        
        # Basic wall kick tests
        kick_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (-2, 0), (2, 0)]
        for kx, ky in kick_offsets:
            if not self._check_collision(self.current_piece['pos_x'] + kx, self.current_piece['pos_y'] + ky, rotated_shape):
                self.current_piece['pos_x'] += kx
                self.current_piece['pos_y'] += ky
                self.current_piece['shape'] = rotated_shape
                return

    def _lock_piece(self):
        # sfx: lock_sound
        for dx, dy in self.current_piece['shape']:
            x, y = self.current_piece['pos_x'] + dx, self.current_piece['pos_y'] + dy
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[y, x] = self.current_piece['color_idx']
        
        self.current_piece = None
        self._spawn_piece()

    def _find_full_lines(self):
        return [y for y in range(self.GRID_HEIGHT) if np.all(self.grid[y, :] > 0)]
        
    def _perform_line_clear(self):
        lines_to_clear = sorted(self.flashing_lines, reverse=True)
        for y in lines_to_clear:
            self.grid = np.delete(self.grid, y, axis=0)
            new_row = np.zeros((1, self.GRID_WIDTH), dtype=int)
            self.grid = np.vstack([new_row, self.grid])
        self.flashing_lines = []

    # --- Gymnasium Info ---
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Controls ---
    # pip install keyboard
    try:
        import keyboard
        print("\n" + "="*30)
        print("MANUAL PLAY ENABLED")
        print(env.user_guide)
        print("Press 'q' to quit.")
        print("="*30 + "\n")

        # Pygame window for human play
        human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Dropper")
        
        terminated = False
        while not terminated:
            # Map keyboard to MultiDiscrete action
            movement = 0 # No-op
            if keyboard.is_pressed('up arrow'): movement = 1
            elif keyboard.is_pressed('down arrow'): movement = 2
            elif keyboard.is_pressed('left arrow'): movement = 3
            elif keyboard.is_pressed('right arrow'): movement = 4
            
            space_held = 1 if keyboard.is_pressed('space') else 0
            shift_held = 1 if keyboard.is_pressed('shift') else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)

            # Render for human
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            human_screen.blit(surf, (0, 0))
            pygame.display.flip()

            if keyboard.is_pressed('q'):
                break

            env.clock.tick(15) # Control game speed for human play

    except ImportError:
        print("\n'keyboard' library not found. Running a short random agent test instead.")
        print("Install it with: pip install keyboard\n")
        
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Episode finished after {info['steps']} steps with score {info['score']}.")
                obs, info = env.reset()

    env.close()