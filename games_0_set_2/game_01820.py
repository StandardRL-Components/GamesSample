
# Generated: 2025-08-27T18:23:16.417799
# Source Brief: brief_01820.md
# Brief Index: 1820

        
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
        "Controls: ←→ to move shape, ↑↓ to rotate. Press space to drop the shape and end your turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A spatial reasoning puzzle. Place falling shapes to perfectly fill the target area before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 15
        self.CELL_SIZE = 24
        self.MAX_MOVES = 20
        self.TARGET_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.TARGET_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_OUTLINE = (200, 200, 220)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_GHOST = (255, 255, 255, 50)
        self.SHAPE_COLORS = [
            (239, 131, 84),  # Orange
            (25, 130, 196),  # Blue
            (148, 210, 133), # Green
            (255, 202, 58),  # Yellow
            (155, 89, 182),  # Purple
            (231, 76, 60),   # Red
            (26, 188, 156),  # Teal
        ]
        self.FILLED_COLORS = [pygame.Color(69, 170, 242).lerp(pygame.Color(20, 25, 40), i / self.MAX_MOVES) for i in range(self.MAX_MOVES + 1)]

        # Shapes (Tetrominos) - defined by offsets from a pivot point
        self.SHAPES = {
            'T': [[(0, 0), (-1, 0), (1, 0), (0, -1)], [(0, 0), (0, -1), (0, 1), (1, 0)], [(0, 0), (-1, 0), (1, 0), (0, 1)], [(0, 0), (0, -1), (0, 1), (-1, 0)]],
            'I': [[(0, 0), (-1, 0), (1, 0), (2, 0)], [(0, 0), (0, -1), (0, 1), (0, 2)]],
            'L': [[(0, 0), (-1, 0), (1, 0), (1, -1)], [(0, 0), (0, -1), (0, 1), (1, 1)], [(0, 0), (-1, 0), (1, 0), (-1, 1)], [(0, 0), (0, -1), (0, 1), (-1, -1)]],
            'J': [[(0, 0), (-1, 0), (1, 0), (-1, -1)], [(0, 0), (0, -1), (0, 1), (1, -1)], [(0, 0), (-1, 0), (1, 0), (1, 1)], [(0, 0), (0, -1), (0, 1), (-1, 1)]],
            'S': [[(0, 0), (-1, 0), (0, -1), (1, -1)], [(0, 0), (0, -1), (1, 0), (1, 1)]],
            'Z': [[(0, 0), (1, 0), (0, -1), (-1, -1)], [(0, 0), (0, 1), (1, 0), (1, -1)]],
            'O': [[(0, 0), (1, 0), (0, 1), (1, 1)]]
        }
        self.SHAPE_KEYS = list(self.SHAPES.keys())

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # Initialize state variables
        self.np_random = None
        self.grid = None
        self.current_shape_key = None
        self.current_shape_color_idx = None
        self.current_shape_pos = None
        self.current_shape_rot = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        self._spawn_new_shape()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        space_held = action[1] == 1
        
        if space_held:
            # Drop action consumes a turn
            reward = self._drop_shape()
            self.moves_left -= 1
            if not self.game_over:
                self._spawn_new_shape()
        else:
            # Movement/rotation does not consume a turn
            self._handle_movement(movement)
        
        self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            # Assign terminal reward only once
            is_win = np.all(self.grid > 0)
            reward += 100 if is_win else -10
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        # 0=none, 1=up(rot_cw), 2=down(rot_ccw), 3=left, 4=right
        if movement == 3: # Move Left
            new_pos = [self.current_shape_pos[0] - 1, self.current_shape_pos[1]]
            if self._is_valid_position(self.current_shape_key, self.current_shape_rot, new_pos):
                self.current_shape_pos = new_pos
        elif movement == 4: # Move Right
            new_pos = [self.current_shape_pos[0] + 1, self.current_shape_pos[1]]
            if self._is_valid_position(self.current_shape_key, self.current_shape_rot, new_pos):
                self.current_shape_pos = new_pos
        elif movement == 1: # Rotate Clockwise
            num_rotations = len(self.SHAPES[self.current_shape_key])
            new_rot = (self.current_shape_rot + 1) % num_rotations
            if self._is_valid_position(self.current_shape_key, new_rot, self.current_shape_pos):
                self.current_shape_rot = new_rot
        elif movement == 2: # Rotate Counter-Clockwise
            num_rotations = len(self.SHAPES[self.current_shape_key])
            new_rot = (self.current_shape_rot - 1 + num_rotations) % num_rotations
            if self._is_valid_position(self.current_shape_key, new_rot, self.current_shape_pos):
                self.current_shape_rot = new_rot

    def _drop_shape(self):
        # 1. Find landing spot
        landing_y = self._get_ghost_y()
        landing_pos = [self.current_shape_pos[0], landing_y]
        
        # 2. Stamp shape onto grid
        shape_coords = self._get_shape_coords(self.current_shape_key, self.current_shape_rot, landing_pos)
        
        cells_filled = 0
        for x, y in shape_coords:
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[x, y] = self.moves_left
                cells_filled += 1
        
        # 3. Calculate reward
        reward = cells_filled # +1 per cell filled

        lines_cleared, self.grid = self._check_and_clear_lines()
        reward += lines_cleared * 5 # +5 per cleared line

        hole_penalty = self._calculate_holes() * 0.2
        reward -= hole_penalty

        self.score += reward
        return reward

    def _spawn_new_shape(self):
        self.current_shape_key = self.SHAPE_KEYS[self.np_random.integers(0, len(self.SHAPE_KEYS))]
        self.current_shape_color_idx = self.np_random.integers(0, len(self.SHAPE_COLORS))
        self.current_shape_rot = 0
        self.current_shape_pos = [self.GRID_WIDTH // 2, 1] # Start near top-middle

        # If spawn position is invalid, game over
        if not self._is_valid_position(self.current_shape_key, self.current_shape_rot, self.current_shape_pos):
             self.game_over = True

    def _get_shape_coords(self, shape_key, rotation, pos):
        shape_offsets = self.SHAPES[shape_key][rotation]
        return [(pos[0] + dx, pos[1] + dy) for dx, dy in shape_offsets]

    def _is_valid_position(self, shape_key, rotation, pos):
        coords = self._get_shape_coords(shape_key, rotation, pos)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False # Out of bounds
            if self.grid[x, y] > 0:
                return False # Collision with existing block
        return True

    def _get_ghost_y(self):
        y = self.current_shape_pos[1]
        while self._is_valid_position(self.current_shape_key, self.current_shape_rot, [self.current_shape_pos[0], y + 1]):
            y += 1
        return y

    def _check_and_clear_lines(self):
        lines_cleared = 0
        new_grid = np.zeros_like(self.grid)
        new_y = self.GRID_HEIGHT - 1
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if np.all(self.grid[:, y] > 0):
                lines_cleared += 1
            else:
                new_grid[:, new_y] = self.grid[:, y]
                new_y -= 1
        return lines_cleared, new_grid
    
    def _calculate_holes(self):
        holes = 0
        for x in range(self.GRID_WIDTH):
            col_has_block = False
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] > 0:
                    col_has_block = True
                elif col_has_block:
                    holes += 1 # An empty space below a filled one in the same column
        return holes

    def _check_termination(self):
        if self.moves_left <= 0:
            return True
        if np.all(self.grid > 0): # Grid completely full
            return True
        return self.game_over # Spawn was blocked

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.TARGET_X, self.TARGET_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw placed blocks
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] > 0:
                    move_idx = int(self.grid[x, y])
                    color = self.FILLED_COLORS[min(move_idx, len(self.FILLED_COLORS)-1)]
                    self._draw_cell(x, y, color)

        # Draw ghost piece
        if not self.game_over:
            ghost_y = self._get_ghost_y()
            ghost_pos = [self.current_shape_pos[0], ghost_y]
            ghost_coords = self._get_shape_coords(self.current_shape_key, self.current_shape_rot, ghost_pos)
            for x, y in ghost_coords:
                self._draw_cell(x, y, self.COLOR_GHOST, is_ghost=True)

        # Draw current shape
        if not self.game_over:
            shape_coords = self._get_shape_coords(self.current_shape_key, self.current_shape_rot, self.current_shape_pos)
            color = self.SHAPE_COLORS[self.current_shape_color_idx]
            for x, y in shape_coords:
                self._draw_cell(x, y, color)

        # Draw grid outline
        pygame.draw.rect(self.screen, self.COLOR_OUTLINE, grid_rect, 2)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        px, py = self.TARGET_X + grid_x * self.CELL_SIZE, self.TARGET_Y + grid_y * self.CELL_SIZE
        cell_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        if is_ghost:
            # Special rendering for ghost
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect(), 0)
            self.screen.blit(s, (px, py))
        else:
            # Filled cell with border
            pygame.draw.rect(self.screen, color, cell_rect)
            border_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, border_color, cell_rect, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Moves Left
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 20, 20))

        if self.game_over:
            is_win = np.all(self.grid > 0)
            msg_text = "AREA FILLED!" if is_win else "OUT OF MOVES"
            msg_color = (100, 255, 100) if is_win else (255, 100, 100)
            
            rendered_msg = self.font_msg.render(msg_text, True, msg_color)
            msg_rect = rendered_msg.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(rendered_msg, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "is_win": np.all(self.grid > 0) if self.game_over else False
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Use a different screen for display to not interfere with the env's headless one
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Geometric Filler")
    
    action = [0, 0, 0] # No-op, no space, no shift
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # Map keyboard events to the MultiDiscrete action space
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Key presses for movement are momentary
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Add a reset key for convenience
                    obs, info = env.reset()

        # Step the environment with the current action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")
        
        # Reset momentary actions after they are processed
        action = [0, 0, 0]

        # Render the observation to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for playability
        
    print("Game Over!")
    pygame.time.wait(2000)
    env.close()