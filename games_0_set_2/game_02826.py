
# Generated: 2025-08-27T21:33:13.061353
# Source Brief: brief_02826.md
# Brief Index: 2826

        
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
        "Controls: ←/→ to move, ↑/↓ to rotate. Space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced block-stacking game. Clear lines to score and survive for 60 seconds."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game world
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    CELL_SIZE = 18
    GRID_LINE_WIDTH = 1
    
    # Screen dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Play area position
    PLAY_AREA_X = (SCREEN_WIDTH - (GRID_WIDTH * CELL_SIZE + (GRID_WIDTH - 1) * GRID_LINE_WIDTH)) // 2
    PLAY_AREA_Y = (SCREEN_HEIGHT - (GRID_HEIGHT * CELL_SIZE + (GRID_HEIGHT - 1) * GRID_LINE_WIDTH)) // 2
    
    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_BG = (50, 50, 70)
    COLOR_TIME_BAR = (100, 200, 255)
    COLOR_TIME_BAR_BG = (60, 60, 80)
    COLOR_WHITE = (255, 255, 255)

    # Tetromino shapes and colors
    TETROMINOES = [
        ([[1, 1], [1, 1]], (255, 255, 0)),         # O (Yellow)
        ([[0, 1, 0], [1, 1, 1], [0, 0, 0]], (160, 0, 255)), # T (Purple)
        ([[1, 1, 0], [0, 1, 1], [0, 0, 0]], (255, 0, 0)),   # Z (Red)
        ([[0, 1, 1], [1, 1, 0], [0, 0, 0]], (0, 255, 0)),   # S (Green)
        ([[0, 0, 1], [1, 1, 1], [0, 0, 0]], (0, 0, 255)),   # J (Blue)
        ([[1, 0, 0], [1, 1, 1], [0, 0, 0]], (255, 165, 0)), # L (Orange)
        ([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], (0, 191, 255)), # I (Cyan)
        ([[1, 1, 1]], (255, 105, 180)), # 3-bar (Pink)
        ([[1]], (200, 200, 200)), # 1-block (Gray)
        ([[1, 0, 1], [0, 1, 0], [1, 0, 1]], (128, 0, 128)), # X-shape (Indigo)
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # State variables initialized in reset
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.fall_time = 0
        self.fall_speed = 0
        self.total_time_s = 0
        
        self.prev_action = None
        self.move_timer = 0
        self.move_delay = 5 # frames
        
        self.particles = []
        
        # Initialize state
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        
        self.fall_time = 0
        self.fall_speed = 0.5  # seconds per grid cell
        self.total_time_s = 0
        
        self.prev_action = self.action_space.sample() * 0 # all zeros
        self.move_timer = 0
        
        self.particles = []
        
        # Generate initial pieces
        self._spawn_new_piece()
        self._spawn_new_piece() # First one becomes current, second becomes next

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        # --- Update Timers & State ---
        delta_time = 1 / 30.0 # Assume 30 FPS
        self.total_time_s += delta_time
        self.steps += 1
        
        if self.game_over:
            terminated = True
        else:
            # --- Handle Input ---
            self._handle_input(action)
            
            # --- Game Logic ---
            self.fall_time += delta_time
            
            # Difficulty scaling
            self.fall_speed = max(0.1, 0.5 - (self.total_time_s / 10.0) * 0.05)
            
            if self.fall_time >= self.fall_speed:
                self.fall_time = 0
                if not self._move_piece(0, 1):
                    self._lock_piece()
                    reward += 0.1 # Small reward for placing a piece
                    
                    lines_cleared, clear_reward = self._clear_lines()
                    reward += clear_reward
                    if lines_cleared > 0:
                        # sfx: line clear
                        pass
                    
                    self._spawn_new_piece()
                    if self.game_over:
                        # sfx: game over
                        reward = -100.0
                        terminated = True
            
            # --- Update particles ---
            self._update_particles(delta_time)
            
            # --- Check Termination Conditions ---
            if not terminated:
                if self.total_time_s >= 60:
                    self.game_won = True
                    reward = 100.0
                    terminated = True
                elif self.steps >= 1800: # 60s @ 30fps
                    # Time ran out but didn't officially win (edge case)
                    terminated = True


        self.prev_action = action
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        move_action = action[0]
        hard_drop_action = action[1]
        
        # --- Rotation (on press) ---
        # 1 = up -> rotate CW
        if move_action == 1 and self.prev_action[0] != 1:
            self._rotate_piece(1)
        # 2 = down -> rotate CCW
        elif move_action == 2 and self.prev_action[0] != 2:
            self._rotate_piece(-1)

        # --- Movement (with auto-repeat) ---
        if move_action in [3, 4]:
            if move_action != self.prev_action[0]:
                self.move_timer = 0 # Reset timer for new direction
                if move_action == 3: self._move_piece(-1, 0) # Left
                elif move_action == 4: self._move_piece(1, 0) # Right
            else:
                self.move_timer += 1
                if self.move_timer > self.move_delay:
                    if move_action == 3: self._move_piece(-1, 0)
                    elif move_action == 4: self._move_piece(1, 0)
        else:
            self.move_timer = 0
        
        # --- Hard Drop (on press) ---
        if hard_drop_action == 1 and self.prev_action[1] == 0:
            # sfx: hard drop
            moved_dist = 0
            while self._move_piece(0, 1):
                moved_dist += 1
            
            self._lock_piece()
            lines_cleared, clear_reward = self._clear_lines()
            # Reward for hard drop is implicit in quicker piece placement
            
            if lines_cleared > 0:
                # sfx: line clear
                pass
            
            self._spawn_new_piece()
            if self.game_over:
                # sfx: game over
                pass

    def _spawn_new_piece(self):
        if self.next_piece is None:
            # First piece
            idx = self.np_random.integers(0, len(self.TETROMINOES))
            shape, color = self.TETROMINOES[idx]
            self.current_piece = {
                "shape": np.array(shape), "color_idx": idx, "x": 0, "y": 0
            }
        else:
            self.current_piece = self.next_piece

        # Position new piece at top-center
        self.current_piece["x"] = self.GRID_WIDTH // 2 - len(self.current_piece["shape"][0]) // 2
        self.current_piece["y"] = 0
        
        # Generate next piece
        idx = self.np_random.integers(0, len(self.TETROMINOES))
        shape, color = self.TETROMINOES[idx]
        self.next_piece = {
            "shape": np.array(shape), "color_idx": idx, "x": 0, "y": 0
        }
        
        # Check for game over
        if not self._is_valid_position(self.current_piece["shape"], self.current_piece["x"], self.current_piece["y"]):
            self.game_over = True
            self.current_piece = None # Don't draw a piece if game is over

    def _is_valid_position(self, shape, grid_x, grid_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    actual_x = grid_x + x
                    actual_y = grid_y + y
                    if not (0 <= actual_x < self.GRID_WIDTH and 0 <= actual_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[actual_y, actual_x] != 0:
                        return False
        return True

    def _move_piece(self, dx, dy):
        if self.current_piece is None: return False
        
        new_x = self.current_piece["x"] + dx
        new_y = self.current_piece["y"] + dy
        if self._is_valid_position(self.current_piece["shape"], new_x, new_y):
            self.current_piece["x"] = new_x
            self.current_piece["y"] = new_y
            return True
        return False

    def _rotate_piece(self, direction):
        if self.current_piece is None: return
        
        # sfx: rotate
        rotated_shape = np.rot90(self.current_piece["shape"], k=-direction)
        
        # Basic wall kick
        for offset in [0, 1, -1, 2, -2]:
            if self._is_valid_position(rotated_shape, self.current_piece["x"] + offset, self.current_piece["y"]):
                self.current_piece["shape"] = rotated_shape
                self.current_piece["x"] += offset
                break

    def _lock_piece(self):
        if self.current_piece is None: return
        
        # sfx: lock
        shape = self.current_piece["shape"]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + x
                    grid_y = self.current_piece["y"] + y
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece["color_idx"] + 1

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y] > 0):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0, 0

        for y in lines_to_clear:
            self.grid[y] = -1 # Mark for flashing animation
            # Create particles
            for i in range(20):
                self.particles.append({
                    "x": self.PLAY_AREA_X + self.np_random.random() * self.GRID_WIDTH * (self.CELL_SIZE + self.GRID_LINE_WIDTH),
                    "y": self.PLAY_AREA_Y + y * (self.CELL_SIZE + self.GRID_LINE_WIDTH) + self.CELL_SIZE / 2,
                    "vx": (self.np_random.random() - 0.5) * 150,
                    "vy": (self.np_random.random() - 0.5) * 150,
                    "life": 0.5,
                    "size": self.np_random.integers(2, 6),
                    "color": self.COLOR_WHITE
                })
        
        num_cleared = len(lines_to_clear)
        reward_map = {1: 1, 2: 3, 3: 5, 4: 10}
        reward = reward_map.get(num_cleared, 0)
        self.score += reward * 100 # Scale score for display

        # Shift down rows
        new_grid = np.zeros_like(self.grid)
        new_y = self.GRID_HEIGHT - 1
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if y not in lines_to_clear:
                new_grid[new_y] = self.grid[y]
                new_y -= 1
        self.grid = new_grid
        
        return num_cleared, reward

    def _update_particles(self, dt):
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw play area background
        play_area_width = self.GRID_WIDTH * (self.CELL_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH
        play_area_height = self.GRID_HEIGHT * (self.CELL_SIZE + self.GRID_LINE_WIDTH) - self.GRID_LINE_WIDTH
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X, self.PLAY_AREA_Y, play_area_width, play_area_height))

        # Draw locked blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx > 0:
                    _, color = self.TETROMINOES[color_idx - 1]
                    self._draw_block(x, y, color)
                elif color_idx == -1: # Flashing cleared line
                    self._draw_block(x, y, self.COLOR_WHITE)

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self.current_piece["y"]
            while self._is_valid_position(self.current_piece["shape"], self.current_piece["x"], ghost_y + 1):
                ghost_y += 1
            
            shape = self.current_piece["shape"]
            _, color = self.TETROMINOES[self.current_piece["color_idx"]]
            for y_off, row in enumerate(shape):
                for x_off, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece["x"] + x_off, ghost_y + y_off, color, is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            shape = self.current_piece["shape"]
            _, color = self.TETROMINOES[self.current_piece["color_idx"]]
            for y_off, row in enumerate(shape):
                for x_off, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece["x"] + x_off, self.current_piece["y"] + y_off, color)
        
        # Draw particles
        for p in self.particles:
            size = max(0, int(p["size"] * (p["life"] / 0.5)))
            pygame.draw.rect(self.screen, p["color"], (int(p["x"]), int(p["y"]), size, size))

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False):
        px = self.PLAY_AREA_X + grid_x * (self.CELL_SIZE + self.GRID_LINE_WIDTH)
        py = self.PLAY_AREA_Y + grid_y * (self.CELL_SIZE + self.GRID_LINE_WIDTH)
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE), 2)
        else:
            main_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, color, main_rect)
            
            # 3D effect
            darker_color = tuple(max(0, c - 50) for c in color)
            lighter_color = tuple(min(255, c + 50) for c in color)
            
            pygame.draw.line(self.screen, lighter_color, main_rect.topleft, main_rect.topright, 1)
            pygame.draw.line(self.screen, lighter_color, main_rect.topleft, main_rect.bottomleft, 1)
            pygame.draw.line(self.screen, darker_color, main_rect.bottomleft, main_rect.bottomright, 1)
            pygame.draw.line(self.screen, darker_color, main_rect.topright, main_rect.bottomright, 1)

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # --- Time Bar ---
        time_bar_width = self.SCREEN_WIDTH - 40
        time_bar_height = 20
        time_progress = max(0, 1.0 - (self.total_time_s / 60.0))
        
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR_BG, (20, self.SCREEN_HEIGHT - 40, time_bar_width, time_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIME_BAR, (20, self.SCREEN_HEIGHT - 40, time_bar_width * time_progress, time_bar_height))
        time_text = self.font_small.render("TIME", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (25, self.SCREEN_HEIGHT - 38))

        # --- Next Piece Preview ---
        preview_x = self.PLAY_AREA_X + self.GRID_WIDTH * (self.CELL_SIZE + self.GRID_LINE_WIDTH) + 20
        preview_y = self.PLAY_AREA_Y
        
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (preview_x, preview_y))
        
        preview_box_size = 5 * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (preview_x, preview_y + 30, preview_box_size, preview_box_size))
        
        if self.next_piece:
            shape = self.next_piece["shape"]
            _, color = self.TETROMINOES[self.next_piece["color_idx"]]
            
            shape_w = len(shape[0]) * self.CELL_SIZE
            shape_h = len(shape) * self.CELL_SIZE
            
            start_x = preview_x + (preview_box_size - shape_w) // 2
            start_y = preview_y + 30 + (preview_box_size - shape_h) // 2
            
            for y_off, row in enumerate(shape):
                for x_off, cell in enumerate(row):
                    if cell:
                        px = start_x + x_off * self.CELL_SIZE
                        py = start_y + y_off * self.CELL_SIZE
                        pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE))

        # --- Game Over / Win Text ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_main.render("GAME OVER", True, (255, 50, 50))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)
        elif self.game_won:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_main.render("YOU WIN!", True, (50, 255, 50))
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_survived": self.total_time_s,
            "is_game_over": self.game_over,
            "is_win": self.game_won
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
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    terminated = False
    
    # Set up Pygame window for human play
    pygame.display.set_caption("Block Stacker")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # No-op
    
    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Key mapping for human play ---
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        else:
            action[0] = 0

        # Space (hard drop)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift (unused)
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    print(f"Final Info: {info}")
    env.close()