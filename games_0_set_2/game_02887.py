
# Generated: 2025-08-28T06:19:58.903960
# Source Brief: brief_02887.md
# Brief Index: 2887

        
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
        "Controls: ←/→ to move, ↑ to rotate clockwise, ↓ for soft drop. "
        "Hold Shift to rotate counter-clockwise, press Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, grid-based puzzle game. Manipulate falling blocks to complete horizontal lines. "
        "Clear 10 lines to win, but don't let the stack reach the top!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 20
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = 0

        self.WIN_CONDITION_LINES = 10
        self.MAX_STEPS = 2500

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_FLASH = (255, 255, 255)
        self.PIECE_COLORS = [
            (0, 0, 0),  # 0 is empty
            (255, 87, 87),   # I piece (Red)
            (87, 255, 87),   # J piece (Green)
            (87, 87, 255),   # L piece (Blue)
            (255, 255, 87),  # O piece (Yellow)
            (255, 87, 255),  # S piece (Magenta)
            (87, 255, 255),  # T piece (Cyan)
            (255, 165, 0),   # Z piece (Orange)
        ]

        # Tetromino shapes
        self.TETROMINOS = [
            [[1, 1, 1, 1]],  # I
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 1], [1, 1]],  # O
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 1, 1], [0, 1, 0]],  # T
            [[1, 1, 0], [0, 1, 1]],  # Z
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 32, bold=True)

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece_idx = None
        self.fall_time = None
        self.fall_speed = None
        self.score = None
        self.lines_cleared = None
        self.steps = None
        self.game_over = None
        self.line_clear_animation = None
        
        self.reset()
        self.validate_implementation()
    
    def _new_piece(self):
        self.current_piece = {
            "shape_idx": self.next_piece_idx,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 1,
            "y": 0
        }
        self.next_piece_idx = self.np_random.integers(0, len(self.TETROMINOS))
        
        if self._check_collision(self.current_piece, (0, 0)):
            self.game_over = True

    def _get_current_shape(self):
        shape = self.TETROMINOS[self.current_piece["shape_idx"]]
        for _ in range(self.current_piece["rotation"]):
            shape = list(zip(*shape[::-1]))
        return shape

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.next_piece_idx = self.np_random.integers(0, len(self.TETROMINOS))
        self._new_piece()
        
        self.fall_time = 0
        self.fall_speed = 0.5 # Seconds per grid cell
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.line_clear_animation = {"timer": 0, "rows": []}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.01 # Small reward for surviving
        
        if self.game_over:
            return self._get_observation(), -100, True, False, self._get_info()

        # Handle line clear animation delay
        if self.line_clear_animation["timer"] > 0:
            self.line_clear_animation["timer"] -= 1
            if self.line_clear_animation["timer"] == 0:
                self._perform_line_clear()
            return self._get_observation(), 0, False, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling ---
        # Actions are processed once per step
        if space_held: # Hard drop
            drop_dist = 0
            while not self._check_collision(self.current_piece, (0, drop_dist + 1)):
                drop_dist += 1
            self.current_piece["y"] += drop_dist
            reward += 0.01 * drop_dist # Small reward for speed
            self._lock_piece()
            reward += self._process_lock()
        else:
            # Horizontal Movement
            if movement == 3: # Left
                if not self._check_collision(self.current_piece, (-1, 0)): self.current_piece["x"] -= 1
            elif movement == 4: # Right
                if not self._check_collision(self.current_piece, (1, 0)): self.current_piece["x"] += 1
            
            # Rotation
            if movement == 1: # Clockwise
                self._rotate_piece(1)
            if shift_held: # Counter-clockwise
                self._rotate_piece(-1)
            
            # Soft Drop
            if movement == 2:
                self.fall_time += self.fall_speed # Accelerate drop

            # --- Gravity ---
            self.fall_time += self.clock.tick(30) / 1000.0
            if self.fall_time > self.fall_speed:
                self.fall_time = 0
                if not self._check_collision(self.current_piece, (0, 1)):
                    self.current_piece["y"] += 1
                else:
                    self._lock_piece()
                    reward += self._process_lock()

        terminated = self._check_termination()
        if terminated and not self.game_over: # Win condition
            reward = 100
        elif self.game_over: # Lose condition
            reward = -100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_lock(self):
        # This is called after a piece is locked
        reward = 0
        
        # Calculate holes created
        shape = self._get_current_shape()
        holes = 0
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    check_y = self.current_piece["y"] + r_idx + 1
                    check_x = self.current_piece["x"] + c_idx
                    while check_y < self.GRID_HEIGHT and self.grid[check_y][check_x] == 0:
                        holes += 1
                        check_y += 1
        reward -= holes * 0.5

        # Check for line clears
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            self.line_clear_animation = {"timer": 6, "rows": lines_to_clear} # 6 frames = 0.2s at 30fps
            # Rewards and score are given when animation finishes
            reward_map = {1: 1, 2: 2, 3: 3, 4: 4}
            score_map = {1: 100, 2: 300, 3: 500, 4: 800}
            num_lines = len(lines_to_clear)
            reward += reward_map.get(num_lines, 0)
            self.pending_score = score_map.get(num_lines, 0)
            self.pending_lines = num_lines
        else:
            self._new_piece() # No lines cleared, spawn next piece immediately

        return reward

    def _perform_line_clear(self):
        rows = self.line_clear_animation["rows"]
        for r in sorted(rows, reverse=True):
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        
        self.score += self.pending_score
        self.lines_cleared += self.pending_lines
        self.fall_speed = max(0.1, 0.5 - (self.lines_cleared // 2) * 0.05)
        
        self.pending_score = 0
        self.pending_lines = 0
        self._new_piece()

    def _lock_piece(self):
        shape = self._get_current_shape()
        color_idx = self.current_piece["shape_idx"] + 1
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + c_idx
                    grid_y = self.current_piece["y"] + r_idx
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = color_idx
        # sfx: block_lock.wav

    def _rotate_piece(self, direction):
        original_rotation = self.current_piece["rotation"]
        self.current_piece["rotation"] = (self.current_piece["rotation"] + direction) % 4
        
        # Wall kick
        offset = 0
        if self._check_collision(self.current_piece, (0, 0)):
            shape = self._get_current_shape()
            width = len(shape[0])
            if self.current_piece["x"] < 0: offset = -self.current_piece["x"]
            elif self.current_piece["x"] + width > self.GRID_WIDTH: offset = self.GRID_WIDTH - (self.current_piece["x"] + width)
            
            if self._check_collision(self.current_piece, (offset, 0)):
                self.current_piece["rotation"] = original_rotation # Can't rotate
            else:
                self.current_piece["x"] += offset
                # sfx: rotate.wav

    def _check_collision(self, piece, offset):
        shape = self.TETROMINOS[piece["shape_idx"]]
        for _ in range(piece["rotation"]):
            shape = list(zip(*shape[::-1]))
        
        off_x, off_y = offset
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + c_idx + off_x
                    grid_y = piece["y"] + r_idx + off_y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True
                    if self.grid[grid_y, grid_x] != 0:
                        return True
        return False

    def _check_termination(self):
        return self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}
    
    def _draw_block(self, surface, x, y, color_idx, is_ghost=False):
        color = self.PIECE_COLORS[color_idx]
        if is_ghost:
            color = (color[0] // 4, color[1] // 4, color[2] // 4)

        outer_rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        inner_rect = pygame.Rect(x + 2, y + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)
        
        pygame.draw.rect(surface, tuple(c*0.6 for c in color), outer_rect, 0, 3)
        pygame.draw.rect(surface, color, inner_rect, 0, 3)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_Y), (px, self.GRID_Y + self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, py), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, py))

        # Draw landed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(self.screen, self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.grid[r, c])

        # Draw line clear animation
        if self.line_clear_animation["timer"] > 0:
            alpha = 255 * (0.5 + 0.5 * math.sin(self.line_clear_animation["timer"] * math.pi / 3))
            flash_surface = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_FLASH, alpha))
            for r in self.line_clear_animation["rows"]:
                self.screen.blit(flash_surface, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE))
            # sfx: line_clear.wav

        if not self.game_over:
            # Draw ghost piece
            ghost_piece = self.current_piece.copy()
            drop_dist = 0
            while not self._check_collision(ghost_piece, (0, drop_dist + 1)):
                drop_dist += 1
            ghost_piece["y"] += drop_dist
            shape = self._get_current_shape()
            color_idx = self.current_piece["shape_idx"] + 1
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, 
                                         self.GRID_X + (ghost_piece["x"] + c_idx) * self.CELL_SIZE, 
                                         self.GRID_Y + (ghost_piece["y"] + r_idx) * self.CELL_SIZE, 
                                         color_idx, is_ghost=True)

            # Draw current piece
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, 
                                         self.GRID_X + (self.current_piece["x"] + c_idx) * self.CELL_SIZE, 
                                         self.GRID_Y + (self.current_piece["y"] + r_idx) * self.CELL_SIZE, 
                                         color_idx)

    def _render_ui(self):
        ui_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 40
        
        # Score
        score_text = self.font_title.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, 30))
        score_val = self.font_main.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (ui_x, 65))
        
        # Lines
        lines_text = self.font_title.render("LINES", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (ui_x, 120))
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (ui_x, 155))

        # Next Piece
        next_text = self.font_title.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_x, 210))
        
        next_shape = self.TETROMINOS[self.next_piece_idx]
        next_color = self.next_piece_idx + 1
        start_x = ui_x + (120 - len(next_shape[0]) * self.CELL_SIZE) // 2
        start_y = 255
        for r_idx, row in enumerate(next_shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    self._draw_block(self.screen, start_x + c_idx * self.CELL_SIZE, start_y + r_idx * self.CELL_SIZE, next_color)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            text_surf = self.font_title.render(status_text, True, self.COLOR_FLASH)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()

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
    terminated = False
    
    # Mapping from Pygame keys to MultiDiscrete action
    key_map = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }

    # Pygame setup for human play
    pygame.display.set_caption("Gymnasium Tetris")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    while not terminated:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        for key, act in key_map.items():
            if keys[key]:
                action = np.add(action, np.array(act))
        
        # Clamp movement to one direction at a time for human play
        move_actions = [action[0] == i for i in range(1, 5)]
        if sum(move_actions) > 1:
            if action[0] == 3 or action[0] == 4: # Prioritize left/right
                action[0] = 3 if keys[pygame.K_LEFT] else 4
            else:
                action[0] = 1 if keys[pygame.K_UP] else 2

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Score: {info['score']}, Lines: {info['lines']}")
            pygame.time.wait(2000) # Pause before closing
            
    env.close()