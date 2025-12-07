
# Generated: 2025-08-27T13:48:50.586577
# Source Brief: brief_00492.md
# Brief Index: 492

        
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
        "Controls: ←→ to move, ↑ to rotate, ↓ for soft drop. Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, falling block puzzle. Clear lines to score points before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.SIDE_PANEL_WIDTH = 180

        # Center the playfield
        self.GRID_OFFSET_X = (self.WIDTH - self.SIDE_PANEL_WIDTH - (self.GRID_WIDTH * self.CELL_SIZE)) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - (self.GRID_HEIGHT * self.CELL_SIZE)) // 2

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255, 50)
        self.TETROMINO_COLORS = [
            (239, 131, 84),  # L - Orange
            (64, 134, 239),   # J - Blue
            (124, 203, 113), # S - Green
            (239, 84, 84),   # Z - Red
            (241, 222, 113), # O - Yellow
            (171, 108, 239), # T - Purple
            (108, 220, 239), # I - Cyan
        ]

        # --- Tetromino Shapes (pivot at 0,0) ---
        self.TETROMINOES = [
            [(-1, 0), (0, 0), (1, 0), (1, -1)],    # L
            [(-1, -1), (-1, 0), (0, 0), (1, 0)],   # J
            [(-1, 0), (0, 0), (0, -1), (1, -1)],   # S
            [(-1, -1), (0, -1), (0, 0), (1, 0)],   # Z
            [(0, 0), (1, 0), (0, -1), (1, -1)],    # O
            [(-1, 0), (0, 0), (1, 0), (0, -1)],    # T
            [(-2, 0), (-1, 0), (0, 0), (1, 0)],    # I
        ]

        # --- Gymnasium Spaces ---
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
        self.font_title = pygame.font.Font(None, 48)

        # --- Game State ---
        # These are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0
        self.fall_counter = 0
        self.last_action = np.array([0, 0, 0])
        self.reward_buffer = 0

        self.reset()
        # self.validate_implementation() # Optional: Call for debugging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 30  # Frames per grid cell drop
        self.fall_counter = 0
        self.reward_buffer = 0
        self.last_action = np.array([0, 0, 0])

        self._spawn_piece() # Spawn next piece
        self._spawn_piece() # Spawn current piece

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.reward_buffer = -0.01  # Small penalty per step to encourage speed

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()

        reward = self.reward_buffer
        terminated = self._check_termination()
        self.last_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space, shift = action
        
        # Check for rising edge for single-press actions
        up_pressed = movement == 1 and self.last_action[0] != 1
        space_pressed = space == 1 and self.last_action[1] != 1
        
        # Hard drop takes precedence
        if space_pressed:
            # Sfx: Hard drop sound
            self._hard_drop()
            return

        if up_pressed:
            # Sfx: Rotate sound
            self._rotate_piece()

        if movement == 3:  # Left
            self._move_piece(-1, 0)
        elif movement == 4:  # Right
            self._move_piece(1, 0)
        
        if movement == 2: # Down (soft drop)
            self.fall_counter += 5 # Accelerate fall

    def _update_game_state(self):
        self.fall_counter += 1
        if self.fall_counter >= self.fall_speed:
            self.fall_counter = 0
            
            if not self._move_piece(0, 1): # If cannot move down
                self._lock_piece()
                self._clear_lines()
                self._spawn_piece()
                
                # Difficulty scaling
                self.fall_speed = max(5, 30 - (self.lines_cleared // 10) * 2)

    def _spawn_piece(self):
        self.current_piece = self.next_piece
        
        shape_idx = self.np_random.integers(0, len(self.TETROMINOES))
        self.next_piece = {
            "shape_idx": shape_idx,
            "shape": self.TETROMINOES[shape_idx],
            "color_idx": shape_idx,
            "x": self.GRID_WIDTH // 2,
            "y": 1,
            "rotation": 0,
        }
        
        if self.current_piece and not self._is_valid_position(self.current_piece):
            self.game_over = True
            self.reward_buffer -= 100

    def _move_piece(self, dx, dy):
        if self.current_piece is None: return False
        
        test_piece = self.current_piece.copy()
        test_piece["x"] += dx
        test_piece["y"] += dy

        if self._is_valid_position(test_piece):
            self.current_piece = test_piece
            return True
        return False

    def _rotate_piece(self):
        if self.current_piece is None: return
        if self.current_piece["shape_idx"] == 4: return # 'O' piece doesn't rotate

        test_piece = self.current_piece.copy()
        
        # Standard rotation
        rotated_shape = []
        for x, y in test_piece["shape"]:
            rotated_shape.append((y, -x)) # Rotate 90 degrees clockwise
        test_piece["shape"] = rotated_shape

        # Wall kick checks
        offsets_to_test = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        for ox, oy in offsets_to_test:
            kick_piece = test_piece.copy()
            kick_piece["x"] += ox
            kick_piece["y"] += oy
            if self._is_valid_position(kick_piece):
                self.current_piece = kick_piece
                return

    def _hard_drop(self):
        if self.current_piece is None: return
        
        while self._move_piece(0, 1):
            pass # Keep moving down until it can't
        
        self._lock_piece()
        self._clear_lines()
        self._spawn_piece()
        self.fall_counter = self.fall_speed # Trigger next piece immediately

    def _lock_piece(self):
        if self.current_piece is None: return
        # Sfx: Block lock sound
        for x_off, y_off in self.current_piece["shape"]:
            x = self.current_piece["x"] + x_off
            y = self.current_piece["y"] + y_off
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x, y] = self.current_piece["color_idx"] + 1

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[:, y]):
                lines_to_clear.append(y)
        
        if lines_to_clear:
            # Sfx: Line clear sound
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            
            # Scoring
            score_map = {1: 1, 2: 5, 3: 10, 4: 20}
            self.score += score_map.get(num_cleared, 0) * 10
            self.reward_buffer += score_map.get(num_cleared, 0)
            
            # Remove lines and shift down
            for y in sorted(lines_to_clear, reverse=True):
                self.grid[:, 1:y+1] = self.grid[:, 0:y]
                self.grid[:, 0] = 0

    def _is_valid_position(self, piece):
        for x_off, y_off in piece["shape"]:
            x = piece["x"] + x_off
            y = piece["y"] + y_off
            
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False # Out of bounds
            if self.grid[x, y] != 0:
                return False # Collides with existing block
        return True

    def _get_ghost_piece_y(self):
        if self.current_piece is None: return -1
        
        ghost = self.current_piece.copy()
        while self._is_valid_position(ghost):
            ghost["y"] += 1
        return ghost["y"] - 1

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= 1000:
            self.reward_buffer += 100
            return True
        if self.steps >= 10000:
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
            "lines_cleared": self.lines_cleared,
        }

    def _draw_block(self, surface, x, y, color_idx, alpha=255):
        color = self.TETROMINO_COLORS[color_idx]
        
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Bright inner part
        inner_rect = rect.inflate(-4, -4)
        pygame.draw.rect(surface, color, inner_rect, border_radius=2)

        # Darker border
        border_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(surface, border_color, rect, 2, border_radius=3)

    def _render_game(self):
        # Draw grid background
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        grid_surface.fill(self.COLOR_GRID)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                pygame.draw.rect(grid_surface, self.COLOR_BG, (x*self.CELL_SIZE, y*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)
        
        # Draw locked pieces
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] != 0:
                    color_idx = int(self.grid[x, y] - 1)
                    self._draw_block(grid_surface, x * self.CELL_SIZE, y * self.CELL_SIZE, color_idx)

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_y = self._get_ghost_piece_y()
            for x_off, y_off in self.current_piece["shape"]:
                x = (self.current_piece["x"] + x_off) * self.CELL_SIZE
                y = (ghost_y + y_off) * self.CELL_SIZE
                
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(grid_surface, (200, 200, 220), rect, 2, border_radius=3)
        
        # Draw current piece
        if self.current_piece and not self.game_over:
            for x_off, y_off in self.current_piece["shape"]:
                x = (self.current_piece["x"] + x_off) * self.CELL_SIZE
                y = (self.current_piece["y"] + y_off) * self.CELL_SIZE
                self._draw_block(grid_surface, x, y, self.current_piece["color_idx"])
        
        self.screen.blit(grid_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))
        
        # Draw border around playfield
        playfield_rect = pygame.Rect(self.GRID_OFFSET_X - 2, self.GRID_OFFSET_Y - 2, 
                                     self.GRID_WIDTH * self.CELL_SIZE + 4, self.GRID_HEIGHT * self.CELL_SIZE + 4)
        pygame.draw.rect(self.screen, self.COLOR_GRID, playfield_rect, 2, border_radius=5)

    def _render_ui(self):
        ui_x = self.GRID_OFFSET_X + self.GRID_WIDTH * self.CELL_SIZE + 40
        
        # --- Score Display ---
        score_text = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, self.GRID_OFFSET_Y))
        score_val = self.font_main.render(f"{self.score}", True, self.TETROMINO_COLORS[4])
        self.screen.blit(score_val, (ui_x, self.GRID_OFFSET_Y + 25))

        # --- Lines Display ---
        lines_text = self.font_small.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (ui_x, self.GRID_OFFSET_Y + 80))
        lines_val = self.font_main.render(f"{self.lines_cleared}", True, self.TETROMINO_COLORS[6])
        self.screen.blit(lines_val, (ui_x, self.GRID_OFFSET_Y + 105))
        
        # --- Next Piece Display ---
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_x, self.GRID_OFFSET_Y + 160))
        
        next_box_rect = pygame.Rect(ui_x, self.GRID_OFFSET_Y + 185, 100, 80)
        pygame.draw.rect(self.screen, self.COLOR_GRID, next_box_rect, 0, border_radius=5)
        
        if self.next_piece:
            # Center the piece in the preview box
            shape = self.next_piece["shape"]
            min_x = min(p[0] for p in shape)
            max_x = max(p[0] for p in shape)
            min_y = min(p[1] for p in shape)
            max_y = max(p[1] for p in shape)
            
            shape_width = (max_x - min_x + 1) * self.CELL_SIZE
            shape_height = (max_y - min_y + 1) * self.CELL_SIZE
            
            offset_x = next_box_rect.centerx - shape_width / 2 - min_x * self.CELL_SIZE
            offset_y = next_box_rect.centery - shape_height / 2 - min_y * self.CELL_SIZE

            for x_off, y_off in self.next_piece["shape"]:
                self._draw_block(self.screen, offset_x + x_off * self.CELL_SIZE, offset_y + y_off * self.CELL_SIZE, self.next_piece["color_idx"])
                
        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_title.render("GAME OVER", True, self.TETROMINO_COLORS[3])
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # The environment is auto-reset by some wrappers, but here we do it manually after a key press
            # Or you can just let the loop end
            # running = False 
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()