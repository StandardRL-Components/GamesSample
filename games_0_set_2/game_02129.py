import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ to rotate counter-clockwise. Press space to drop the piece."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Rotate and position falling geometric shapes to completely fill the 6x6 grid in this minimalist puzzle game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 6, 6
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    CELL_SIZE = 50
    GRID_LINE_WIDTH = 2

    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (50, 50, 70)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BAR = (60, 60, 80)
    COLOR_UI_FILL = (80, 120, 255)
    COLOR_GHOST = (255, 255, 255, 50)

    SHAPE_COLORS_BRIGHT = [
        (0, 255, 255),  # I (Cyan)
        (255, 255, 0),  # O (Yellow)
        (160, 0, 255),  # T (Purple)
        (0, 0, 255),    # J (Blue)
        (255, 165, 0),  # L (Orange)
        (0, 255, 0),    # S (Green)
        (255, 0, 0),    # Z (Red)
    ]
    SHAPE_COLORS_DARK = [
        (0, 128, 128),
        (128, 128, 0),
        (80, 0, 128),
        (0, 0, 128),
        (128, 82, 0),
        (0, 128, 0),
        (128, 0, 0),
    ]

    # Shape definitions: rotations for each shape type
    SHAPES = [
        # I
        [[(-1, 0), (0, 0), (1, 0), (2, 0)], [(0, -1), (0, 0), (0, 1), (0, 2)]],
        # O
        [[(0, 0), (1, 0), (0, 1), (1, 1)]],
        # T
        [[(-1, 0), (0, 0), (1, 0), (0, -1)], [(0, -1), (0, 0), (0, 1), (1, 0)], [(-1, 0), (0, 0), (1, 0), (0, 1)], [(0, -1), (0, 0), (0, 1), (-1, 0)]],
        # J
        [[(-1, -1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (1, -1), (0, 0), (0, 1)], [(-1, 0), (0, 0), (1, 0), (1, 1)], [(0, -1), (0, 0), (-1, 1), (0, 1)]],
        # L
        [[(-1, 0), (0, 0), (1, 0), (1, -1)], [(0, -1), (0, 0), (0, 1), (1, 1)], [(-1, 1), (-1, 0), (0, 0), (1, 0)], [(0, -1), (-1, -1), (0, 0), (0, 1)]],
        # S
        [[(-1, 0), (0, 0), (0, -1), (1, -1)], [(0, -1), (0, 0), (1, 0), (1, 1)]],
        # Z
        [[(-1, -1), (0, -1), (0, 0), (1, 0)], [(1, -1), (1, 0), (0, 0), (0, 1)]],
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("sans", 24)
        self.font_title = pygame.font.SysFont("sans", 32, bold=True)

        self.grid_render_pos = (
            (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        )

        # Initialize state variables
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.current_shape_id = 0
        self.current_shape_rot = 0
        self.current_shape_pos = (0, 0)
        self.score = 0
        self.steps = 0
        self.game_over = False
        
        # self.validate_implementation() # This is for internal testing, can be commented out


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False

        self._spawn_shape()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        hard_drop = action[1] == 1
        
        reward = 0
        self.steps += 1
        
        if hard_drop:
            reward = self._execute_hard_drop()
            if not self.game_over:
                self._spawn_shape()
        else:
            self._handle_movement(movement)
        
        # Check for termination conditions
        terminated = self.game_over or self.steps >= 1000
        if self.steps >= 1000:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _execute_hard_drop(self):
        # Find landing position
        ghost_pos = self._get_ghost_pos()
        self.current_shape_pos = ghost_pos
        
        # Lock shape
        self._lock_shape()
        
        # Calculate rewards
        reward = 0.4 # +0.1 for each of the 4 blocks
        
        # Clear lines
        cleared_lines = self._clear_lines()
        reward += cleared_lines * 1.0
        
        # Check for win condition
        if np.all(self.grid > 0):
            reward += 100
            self.game_over = True
        
        return reward

    def _handle_movement(self, movement):
        # 0=none, 1=up(rot_cw), 2=down(rot_ccw), 3=left, 4=right
        if movement == 0: # no-op
            return

        new_pos = self.current_shape_pos
        new_rot = self.current_shape_rot
        
        if movement == 1: # Rotate CW
            new_rot = (self.current_shape_rot + 1) % len(self.SHAPES[self.current_shape_id])
        elif movement == 2: # Rotate CCW
            new_rot = (self.current_shape_rot - 1 + len(self.SHAPES[self.current_shape_id])) % len(self.SHAPES[self.current_shape_id])
        elif movement == 3: # Move Left
            new_pos = (self.current_shape_pos[0] - 1, self.current_shape_pos[1])
        elif movement == 4: # Move Right
            new_pos = (self.current_shape_pos[0] + 1, self.current_shape_pos[1])

        if self._is_valid_position(self.current_shape_id, new_rot, new_pos):
            self.current_shape_pos = new_pos
            self.current_shape_rot = new_rot

    def _spawn_shape(self):
        self.current_shape_id = self.np_random.integers(0, len(self.SHAPES))
        self.current_shape_rot = self.np_random.integers(0, len(self.SHAPES[self.current_shape_id]))
        self.current_shape_pos = (self.GRID_WIDTH // 2, 0)

        # Adjust spawn for I-piece
        if self.current_shape_id == 0:
             self.current_shape_pos = (self.GRID_WIDTH // 2 -1, 0)

        if not self._is_valid_position(self.current_shape_id, self.current_shape_rot, self.current_shape_pos):
            self.game_over = True
            self.score -= 100 # Penalty for losing

    def _get_shape_coords(self, shape_id, rotation, position):
        shape_template = self.SHAPES[shape_id][rotation]
        return [(position[0] + dx, position[1] + dy) for dx, dy in shape_template]

    def _is_valid_position(self, shape_id, rotation, position):
        coords = self._get_shape_coords(shape_id, rotation, position)
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return False
            if self.grid[y, x] > 0:
                return False
        return True

    def _lock_shape(self):
        coords = self._get_shape_coords(self.current_shape_id, self.current_shape_rot, self.current_shape_pos)
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.current_shape_id + 1
        self.score += 4 # Small score for placing a piece

    def _clear_lines(self):
        lines_cleared = 0
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_HEIGHT - 1
        for row_idx in range(self.GRID_HEIGHT - 1, -1, -1):
            if not np.all(self.grid[row_idx, :] > 0):
                new_grid[new_row_idx, :] = self.grid[row_idx, :]
                new_row_idx -= 1
            else:
                lines_cleared += 1
        
        if lines_cleared > 0:
            self.grid = new_grid
            self.score += (10 * lines_cleared) ** 2 # Bonus for line clears
        return lines_cleared

    def _get_ghost_pos(self):
        pos = self.current_shape_pos
        while self._is_valid_position(self.current_shape_id, self.current_shape_rot, (pos[0], pos[1] + 1)):
            pos = (pos[0], pos[1] + 1)
        return pos

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
            "grid_fill": np.count_nonzero(self.grid) / (self.GRID_WIDTH * self.GRID_HEIGHT),
        }

    def _render_game(self):
        gx, gy = self.grid_render_pos
        grid_pixel_width = self.GRID_WIDTH * self.CELL_SIZE
        grid_pixel_height = self.GRID_HEIGHT * self.CELL_SIZE

        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (gx, gy, grid_pixel_width, grid_pixel_height))

        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    shape_id = self.grid[r, c] - 1
                    color = self.SHAPE_COLORS_DARK[shape_id]
                    self._draw_cell(c, r, color)

        if not self.game_over:
            # Draw ghost piece
            ghost_pos = self._get_ghost_pos()
            ghost_coords = self._get_shape_coords(self.current_shape_id, self.current_shape_rot, ghost_pos)
            for x, y in ghost_coords:
                self._draw_cell(x, y, self.COLOR_GHOST, is_ghost=True)

            # Draw active piece
            active_coords = self._get_shape_coords(self.current_shape_id, self.current_shape_rot, self.current_shape_pos)
            color = self.SHAPE_COLORS_BRIGHT[self.current_shape_id]
            for x, y in active_coords:
                self._draw_cell(x, y, color)
        
        # Draw grid lines on top
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (gx + i * self.CELL_SIZE, gy), (gx + i * self.CELL_SIZE, gy + grid_pixel_height), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (gx, gy + i * self.CELL_SIZE), (gx + grid_pixel_width, gy + i * self.CELL_SIZE), self.GRID_LINE_WIDTH)

    def _draw_cell(self, grid_x, grid_y, color, is_ghost=False):
        gx, gy = self.grid_render_pos
        px, py = gx + grid_x * self.CELL_SIZE, gy + grid_y * self.CELL_SIZE
        
        if is_ghost:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
            self.screen.blit(s, (px, py))
        else:
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
            
            # 3D effect
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.line(self.screen, highlight, (px + 2, py + 2), (px + self.CELL_SIZE - 3, py + 2), 2)
            pygame.draw.line(self.screen, highlight, (px + 2, py + 2), (px + 2, py + self.CELL_SIZE - 3), 2)
            pygame.draw.line(self.screen, shadow, (px + self.CELL_SIZE - 3, py + 2), (px + self.CELL_SIZE - 3, py + self.CELL_SIZE - 3), 2)
            pygame.draw.line(self.screen, shadow, (px + 2, py + self.CELL_SIZE - 3), (px + self.CELL_SIZE - 3, py + self.CELL_SIZE - 3), 2)


    def _render_ui(self):
        # Render score
        score_text = self.font_title.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Render fill percentage bar
        bar_width, bar_height = 200, 20
        bar_x, bar_y = self.SCREEN_WIDTH - bar_width - 20, 25
        fill_ratio = np.count_nonzero(self.grid) / (self.GRID_WIDTH * self.GRID_HEIGHT)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_FILL, (bar_x, bar_y, int(bar_width * fill_ratio), bar_height), border_radius=5)
        
        fill_text = self.font_main.render(f"{fill_ratio:.0%}", True, self.COLOR_TEXT)
        text_rect = fill_text.get_rect(center=(bar_x + bar_width/2, bar_y + bar_height/2))
        self.screen.blit(fill_text, text_rect)

        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = np.all(self.grid > 0)
            msg = "GRID COMPLETE!" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            end_text = self.font_title.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
    # For human play
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("GeoFill")
    
    terminated = False
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op
    action[1] = 0
    action[2] = 0

    print("--- GeoFill ---")
    print(env.game_description)
    print(env.user_guide)

    while not terminated:
        # Human input mapping
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                action.fill(0) # Reset action on new key press
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
                elif event.key == pygame.K_r: # Add a reset key for playability
                    obs, info = env.reset()
                    continue
                
                # Execute action on key press
                obs, reward, terminated, truncated, info = env.step(action)
                
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()
    pygame.quit()