
# Generated: 2025-08-28T02:48:08.927746
# Source Brief: brief_01817.md
# Brief Index: 1817

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, Shift to rotate counter-clockwise. "
        "↓ for soft drop, Space for hard drop."
    )

    game_description = (
        "Strategically place falling colored blocks in a grid to clear lines and "
        "achieve the target score before the board fills up."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.PLAY_WIDTH = self.GRID_WIDTH * self.BLOCK_SIZE
        self.PLAY_HEIGHT = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.TOP_LEFT_X = (self.SCREEN_WIDTH - self.PLAY_WIDTH) // 2 - 100
        self.TOP_LEFT_Y = (self.SCREEN_HEIGHT - self.PLAY_HEIGHT) // 2

        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 10
        self.INITIAL_FALL_SPEED = 0.8  # seconds per grid cell drop

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_BORDER = (140, 140, 150)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_GHOST = (255, 255, 255, 50)
        self.PIECE_COLORS = [
            (239, 83, 80),   # Red (Z)
            (102, 187, 106), # Green (S)
            (66, 165, 245),  # Blue (J)
            (255, 167, 38),  # Orange (L)
            (255, 238, 88),  # Yellow (O)
            (26, 188, 156),  # Cyan (I)
            (171, 71, 188),  # Purple (T)
        ]

        # --- Piece Shapes (Tetrominoes) ---
        self.PIECE_SHAPES = [
            [[1, 1, 0], [0, 1, 1]],  # Z
            [[0, 1, 1], [1, 1, 0]],  # S
            [[0, 0, 1], [1, 1, 1]],  # J
            [[1, 0, 0], [1, 1, 1]],  # L
            [[1, 1], [1, 1]],       # O
            [[1, 1, 1, 1]],         # I
            [[0, 1, 0], [1, 1, 1]],  # T
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # --- State Variables ---
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.fall_time = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_space_state = False
        self.clear_animation = {"timer": 0, "lines": []}
        self.rng = None

        self.reset()
        # self.validate_implementation() # Commented out for submission, uncomment for testing

    def _create_piece(self):
        shape_idx = self.rng.integers(0, len(self.PIECE_SHAPES))
        shape = self.PIECE_SHAPES[shape_idx]
        color = self.PIECE_COLORS[shape_idx]
        return {
            "x": self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0,
            "shape": shape,
            "color": color,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.grid = [[(0, 0, 0) for _ in range(self.GRID_HEIGHT)] for _ in range(self.GRID_WIDTH)]
        self.current_piece = self._create_piece()
        self.next_piece = self._create_piece()
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.fall_time = 0
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.last_space_state = False
        self.clear_animation = {"timer": 0, "lines": []}

        if not self._is_valid_position(self.current_piece):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def _is_valid_position(self, piece, offset_x=0, offset_y=0):
        for y, row in enumerate(piece["shape"]):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + x + offset_x
                    grid_y = piece["y"] + y + offset_y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_x][grid_y] != (0, 0, 0):
                        return False
        return True

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Hard drop on space press (rising edge)
        if space_held and not self.last_space_state:
            # sfx: hard_drop.wav
            drop_count = 0
            while self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
                drop_count += 1
            reward += self._lock_piece()
            # reward += drop_count * 0.01 # Small reward for dropping
            self.fall_time = self.fall_speed # Force next piece immediately
        else:
            # Movement
            if movement == 3: # Left
                if self._is_valid_position(self.current_piece, offset_x=-1):
                    self.current_piece["x"] -= 1
            elif movement == 4: # Right
                if self._is_valid_position(self.current_piece, offset_x=1):
                    self.current_piece["x"] += 1
            
            # Soft drop
            if movement == 2: # Down
                self.fall_time += 0.2 # Accelerate fall

            # Rotation
            if movement == 1: # Up for CW
                self._rotate_piece(clockwise=True)
            if shift_held:
                self._rotate_piece(clockwise=False)

        self.last_space_state = space_held
        return reward

    def _rotate_piece(self, clockwise=True):
        shape = self.current_piece["shape"]
        if clockwise:
            new_shape = [list(row) for row in zip(*shape[::-1])]
        else:
            new_shape = [list(row) for row in zip(*shape)][::-1]

        original_x = self.current_piece["x"]
        
        # Wall kick tests
        for offset in [0, -1, 1, -2, 2]:
            self.current_piece["x"] += offset
            if self._is_valid_position({"shape": new_shape, "x": self.current_piece["x"], "y": self.current_piece["y"]}):
                self.current_piece["shape"] = new_shape
                # sfx: rotate.wav
                return
            self.current_piece["x"] = original_x # Reset x for next test

    def _lock_piece(self):
        # sfx: lock_piece.wav
        for y, row in enumerate(self.current_piece["shape"]):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + x
                    grid_y = self.current_piece["y"] + y
                    if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                        self.grid[grid_x][grid_y] = self.current_piece["color"]
        
        reward = 0.1 # Small reward for placing a piece
        reward += self._clear_lines()

        self.current_piece = self.next_piece
        self.next_piece = self._create_piece()

        if not self._is_valid_position(self.current_piece):
            self.game_over = True
        
        return reward

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if all(self.grid[x][y] != (0, 0, 0) for x in range(self.GRID_WIDTH)):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # sfx: line_clear.wav
            self.clear_animation = {"timer": 5, "lines": lines_to_clear} # 5 frames of flash
            for y in sorted(lines_to_clear, reverse=True):
                for move_y in range(y, 0, -1):
                    for x in range(self.GRID_WIDTH):
                        self.grid[x][move_y] = self.grid[x][move_y - 1]
                for x in range(self.GRID_WIDTH):
                    self.grid[x][0] = (0, 0, 0)
            
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            
            # Difficulty scaling
            self.fall_speed = max(0.1, self.INITIAL_FALL_SPEED - (self.lines_cleared // 2) * 0.05)

            # Scoring
            score_map = {1: 10, 2: 30, 3: 50, 4: 100}
            reward_map = {1: 1, 2: 3, 3: 5, 4: 10}
            self.score += score_map.get(num_cleared, 0)
            return reward_map.get(num_cleared, 0)
        return 0

    def step(self, action):
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.clock.tick(30)
        
        if self.clear_animation["timer"] > 0:
            self.clear_animation["timer"] -= 1

        reward = self._handle_input(action)

        # Auto-fall logic
        self.fall_time += self.clock.get_time() / 1000.0
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            if self._is_valid_position(self.current_piece, offset_y=1):
                self.current_piece["y"] += 1
            else:
                reward += self._lock_piece()

        # Check termination conditions
        if self.lines_cleared >= self.WIN_CONDITION_LINES and not self.win:
            self.win = True
            reward += 100
        
        if self.game_over:
            reward -= 100

        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "lines_cleared": self.lines_cleared,
        }

    def _render_game(self):
        # Draw play area border
        pygame.draw.rect(self.screen, self.COLOR_BORDER, 
            (self.TOP_LEFT_X - 2, self.TOP_LEFT_Y - 2, self.PLAY_WIDTH + 4, self.PLAY_HEIGHT + 4), 2)
        
        # Draw grid lines
        for x in range(1, self.GRID_WIDTH):
            pygame.draw.line(self.screen, self.COLOR_GRID, 
                (self.TOP_LEFT_X + x * self.BLOCK_SIZE, self.TOP_LEFT_Y),
                (self.TOP_LEFT_X + x * self.BLOCK_SIZE, self.TOP_LEFT_Y + self.PLAY_HEIGHT))
        for y in range(1, self.GRID_HEIGHT):
            pygame.draw.line(self.screen, self.COLOR_GRID,
                (self.TOP_LEFT_X, self.TOP_LEFT_Y + y * self.BLOCK_SIZE),
                (self.TOP_LEFT_X + self.PLAY_WIDTH, self.TOP_LEFT_Y + y * self.BLOCK_SIZE))

        # Draw locked blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x][y] != (0, 0, 0):
                    self._draw_block(x, y, self.grid[x][y])

        # Draw ghost piece
        if not self.game_over:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, offset_y=1):
                ghost_piece["y"] += 1
            self._draw_piece(ghost_piece, ghost=True)

        # Draw current piece
        if not self.game_over:
            self._draw_piece(self.current_piece)
            
        # Draw line clear flash
        if self.clear_animation["timer"] > 0:
            for y in self.clear_animation["lines"]:
                flash_rect = pygame.Rect(self.TOP_LEFT_X, self.TOP_LEFT_Y + y * self.BLOCK_SIZE, self.PLAY_WIDTH, self.BLOCK_SIZE)
                pygame.draw.rect(self.screen, (255, 255, 255), flash_rect)

    def _draw_piece(self, piece, ghost=False):
        for y, row in enumerate(piece["shape"]):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + x
                    grid_y = piece["y"] + y
                    if ghost:
                        rect = pygame.Rect(
                            self.TOP_LEFT_X + grid_x * self.BLOCK_SIZE,
                            self.TOP_LEFT_Y + grid_y * self.BLOCK_SIZE,
                            self.BLOCK_SIZE, self.BLOCK_SIZE
                        )
                        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2)
                    else:
                        self._draw_block(grid_x, grid_y, piece["color"])

    def _draw_block(self, grid_x, grid_y, color):
        rect = pygame.Rect(
            self.TOP_LEFT_X + grid_x * self.BLOCK_SIZE,
            self.TOP_LEFT_Y + grid_y * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, color, rect)
        # Add a subtle 3D effect
        darker_color = tuple(max(0, c - 40) for c in color)
        lighter_color = tuple(min(255, c + 40) for c in color)
        pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.topright, 1)
        pygame.draw.line(self.screen, lighter_color, rect.topleft, rect.bottomleft, 1)
        pygame.draw.line(self.screen, darker_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, darker_color, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_title.render("SCORE", True, self.COLOR_TEXT)
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.TOP_LEFT_X + self.PLAY_WIDTH + 40, self.TOP_LEFT_Y))
        self.screen.blit(score_val, (self.TOP_LEFT_X + self.PLAY_WIDTH + 40, self.TOP_LEFT_Y + 35))

        # Lines
        lines_text = self.font_title.render("LINES", True, self.COLOR_TEXT)
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.TOP_LEFT_X + self.PLAY_WIDTH + 40, self.TOP_LEFT_Y + 90))
        self.screen.blit(lines_val, (self.TOP_LEFT_X + self.PLAY_WIDTH + 40, self.TOP_LEFT_Y + 125))

        # Next Piece
        next_text = self.font_title.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.TOP_LEFT_X + self.PLAY_WIDTH + 40, self.TOP_LEFT_Y + 180))
        next_box = pygame.Rect(self.TOP_LEFT_X + self.PLAY_WIDTH + 38, self.TOP_LEFT_Y + 218, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, next_box, 2)
        if self.next_piece:
            shape = self.next_piece["shape"]
            w, h = len(shape[0]), len(shape)
            start_x = next_box.centerx - (w * self.BLOCK_SIZE) / 2
            start_y = next_box.centery - (h * self.BLOCK_SIZE) / 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(
                            start_x + x * self.BLOCK_SIZE,
                            start_y + y * self.BLOCK_SIZE,
                            self.BLOCK_SIZE, self.BLOCK_SIZE
                        )
                        pygame.draw.rect(self.screen, self.next_piece["color"], rect)

        # Game Over / Win Text
        if self.game_over:
            self._render_overlay_text("GAME OVER")
        elif self.win:
            self._render_overlay_text("YOU WIN!")

    def _render_overlay_text(self, text):
        overlay = pygame.Surface((self.PLAY_WIDTH, 100), pygame.SRCALPHA)
        overlay.fill((25, 25, 35, 200))
        text_surf = self.font_title.render(text, True, (255, 255, 100))
        text_rect = text_surf.get_rect(center=(self.PLAY_WIDTH / 2, 50))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (self.TOP_LEFT_X, self.TOP_LEFT_Y + self.PLAY_HEIGHT/2 - 50))

    def close(self):
        pygame.quit()

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
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Key mapping for human play
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Main game loop for human play
    running = True
    action = env.action_space.sample() # Start with a random action
    action[0] = 0 # No-op movement
    action[1] = 0 # Space released
    action[2] = 0 # Shift released
    
    # Create a display for human play
    pygame.display.set_caption("Gymnasium Tetris")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    while running:
        # Reset action at the start of each frame
        action[0] = 0 # No-op movement
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                
        # Get key states for continuous actions
        keys = pygame.key.get_pressed()
        
        for key, move_val in key_to_action.items():
            if keys[key]:
                action[0] = move_val
                
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Lines: {info['lines_cleared']}")
            
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("Game Over or Won! Press 'R' to restart.")
            
    env.close()