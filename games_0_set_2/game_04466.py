
# Generated: 2025-08-28T02:29:39.051539
# Source Brief: brief_04466.md
# Brief Index: 4466

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑ to rotate, ↓ to soft drop. Press space to hard drop."
    )

    game_description = (
        "Rotate and drop tetromino-like blocks to clear lines in a fast-paced, grid-based puzzle game."
    )

    auto_advance = True
    
    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_CELL_SIZE = 18
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * GRID_CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * GRID_CELL_SIZE) // 2
    
    MAX_STEPS = 10000
    WIN_CONDITION_LINES = 50

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_WHITE = (255, 255, 255)
    
    TETROMINO_COLORS = [
        (0, 0, 0),          # 0: Empty
        (45, 226, 230),     # 1: I (Cyan)
        (230, 221, 45),     # 2: O (Yellow)
        (177, 45, 230),     # 3: T (Purple)
        (45, 63, 230),      # 4: J (Blue)
        (230, 134, 45),     # 5: L (Orange)
        (71, 230, 45),      # 6: S (Green)
        (230, 45, 52)       # 7: Z (Red)
    ]

    TETROMINO_SHAPES = {
        1: [[1, 1, 1, 1]],  # I
        2: [[1, 1], [1, 1]], # O
        3: [[0, 1, 0], [1, 1, 1]], # T
        4: [[1, 0, 0], [1, 1, 1]], # J
        5: [[0, 0, 1], [1, 1, 1]], # L
        6: [[0, 1, 1], [1, 1, 0]], # S
        7: [[1, 1, 0], [0, 1, 1]], # Z
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        self.reset()
        self.validate_implementation()

    def _create_tetromino(self):
        shape_id = self.np_random.integers(1, len(self.TETROMINO_SHAPES) + 1)
        shape = self.TETROMINO_SHAPES[shape_id]
        return {
            "id": shape_id,
            "shape": np.array(shape, dtype=int),
            "x": self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            "y": 0,
            "color": self.TETROMINO_COLORS[shape_id]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.bag = list(range(1, len(self.TETROMINO_SHAPES) + 1))
        self.np_random.shuffle(self.bag)
        
        self.current_block = self._create_tetromino()
        self.next_block = self._create_tetromino()
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.drop_interval = 1.0  # seconds per grid cell
        self.drop_timer = 0.0
        
        self.last_action_time = { "move": 0, "rotate": 0 }
        self.action_cooldown = 0.12 # seconds

        self.line_clear_animation = [] # list of (row_index, timer)

        if not self._is_valid_position(self.current_block):
             self.game_over = True

        return self._get_observation(), self._get_info()

    def _is_valid_position(self, block, offset_x=0, offset_y=0):
        shape = block["shape"]
        pos_x, pos_y = block["x"] + offset_x, block["y"] + offset_y

        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y, grid_x = pos_y + y, pos_x + x
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_y, grid_x] != 0:
                        return False
        return True

    def _rotate_block(self):
        # sound placeholder: # sfx_rotate.play()
        rotated_shape = np.rot90(self.current_block["shape"], k=-1)
        original_shape = self.current_block["shape"]
        self.current_block["shape"] = rotated_shape
        
        # Wall kick implementation (basic)
        if not self._is_valid_position(self.current_block):
            # Try moving left
            if self._is_valid_position(self.current_block, offset_x=-1):
                self.current_block["x"] -= 1
            # Try moving right
            elif self._is_valid_position(self.current_block, offset_x=1):
                self.current_block["x"] += 1
            # Try moving further right (for I-block)
            elif self._is_valid_position(self.current_block, offset_x=2):
                self.current_block["x"] += 2
            # Revert if no valid position found
            else:
                self.current_block["shape"] = original_shape

    def _lock_block(self):
        # sound placeholder: # sfx_lock.play()
        shape = self.current_block["shape"]
        pos_x, pos_y = self.current_block["x"], self.current_block["y"]

        holes_before = self._count_holes()

        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[pos_y + y, pos_x + x] = self.current_block["id"]

        holes_after = self._count_holes()
        new_holes = holes_after - holes_before
        
        cleared_count = self._clear_lines()

        self.current_block = self.next_block
        self.next_block = self._create_tetromino()

        if not self._is_valid_position(self.current_block):
            self.game_over = True
        
        return cleared_count, new_holes
    
    def _count_holes(self):
        holes = 0
        for col in range(self.GRID_WIDTH):
            found_block = False
            for row in range(self.GRID_HEIGHT):
                if self.grid[row, col] != 0:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def _clear_lines(self):
        rows_to_clear = [r for r, row in enumerate(self.grid) if np.all(row != 0)]
        if not rows_to_clear:
            return 0
        
        # sound placeholder: # sfx_clear.play()
        for row_idx in rows_to_clear:
            self.line_clear_animation.append([row_idx, 0.2]) # 0.2 second animation

        # Create new grid without the cleared lines
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in rows_to_clear:
                new_grid[new_row_idx] = self.grid[r]
                new_row_idx -= 1
        
        self.grid = new_grid
        self.lines_cleared += len(rows_to_clear)

        # Update difficulty
        level = self.lines_cleared // 10
        self.drop_interval = max(0.2, 1.0 - level * 0.05)
        
        return len(rows_to_clear)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty per step to encourage speed
        
        movement, space_pressed, _ = action[0], action[1] == 1, action[2] == 1
        
        time_delta = self.clock.tick(30) / 1000.0
        self.drop_timer += time_delta
        current_time = pygame.time.get_ticks() / 1000.0

        # --- Handle player actions ---
        if space_pressed:
            # Hard drop
            # sound placeholder: # sfx_hard_drop.play()
            drop_dist = 0
            while self._is_valid_position(self.current_block, offset_y=1):
                self.current_block["y"] += 1
                drop_dist += 1
            reward += drop_dist * 0.1
            cleared, holes = self._lock_block()
        else:
            # Horizontal movement
            if movement in [3, 4] and current_time > self.last_action_time["move"] + self.action_cooldown:
                move_dir = -1 if movement == 3 else 1
                if self._is_valid_position(self.current_block, offset_x=move_dir):
                    self.current_block["x"] += move_dir
                    self.last_action_time["move"] = current_time
                    # sound placeholder: # sfx_move.play()
            
            # Rotation
            if movement == 1 and current_time > self.last_action_time["rotate"] + self.action_cooldown:
                self._rotate_block()
                self.last_action_time["rotate"] = current_time

            # Soft drop
            if movement == 2:
                self.drop_timer += 0.1 # Accelerate drop
                if self._is_valid_position(self.current_block, offset_y=1):
                    self.current_block["y"] += 1
                    reward += 0.1
                    self.drop_timer = 0
                else:
                    cleared, holes = self._lock_block()
            
            # --- Handle auto-drop ---
            cleared, holes = 0, 0
            if self.drop_timer >= self.drop_interval:
                self.drop_timer = 0
                if self._is_valid_position(self.current_block, offset_y=1):
                    self.current_block["y"] += 1
                else:
                    cleared, holes = self._lock_block()

        # --- Calculate rewards from locking a piece ---
        if 'cleared' in locals() and cleared > 0:
            reward_map = {1: 1, 2: 3, 3: 5, 4: 10}
            self.score += reward_map[cleared] * 100
            reward += reward_map.get(cleared, 0)
        if 'holes' in locals() and holes > 0:
            reward -= holes * 0.2

        # --- Check for termination conditions ---
        terminated = False
        if self.game_over:
            reward -= 100
            terminated = True
        elif self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
            self.score += 10000
            terminated = True
            self.game_over = True # End game on win
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_block(self, block, offset_x=0, offset_y=0, alpha=255):
        shape = block["shape"]
        pos_x, pos_y = block["x"] + offset_x, block["y"] + offset_y
        color = block["color"]
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    px, py = (pos_x + x) * self.GRID_CELL_SIZE, (pos_y + y) * self.GRID_CELL_SIZE
                    rect = pygame.Rect(self.GRID_X_OFFSET + px, self.GRID_Y_OFFSET + py, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                    
                    if alpha < 255:
                        s = pygame.Surface((self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), pygame.SRCALPHA)
                        s.fill((*color, alpha))
                        self.screen.blit(s, rect.topleft)
                    else:
                        # Main block color
                        pygame.draw.rect(self.screen, color, rect)
                        # Highlight
                        pygame.draw.rect(self.screen, tuple(min(255, c+50) for c in color), rect.inflate(-4, -4))
                        # Border
                        pygame.draw.rect(self.screen, self.COLOR_BG, rect, 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Grid ---
        grid_surface = pygame.Surface((self.GRID_WIDTH * self.GRID_CELL_SIZE, self.GRID_HEIGHT * self.GRID_CELL_SIZE))
        grid_surface.fill(self.COLOR_GRID)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    color = self.TETROMINO_COLORS[self.grid[y, x]]
                    rect = pygame.Rect(x * self.GRID_CELL_SIZE, y * self.GRID_CELL_SIZE, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE)
                    pygame.draw.rect(grid_surface, color, rect)
                    pygame.draw.rect(grid_surface, tuple(min(255, c+50) for c in color), rect.inflate(-4, -4))
                    pygame.draw.rect(grid_surface, self.COLOR_BG, rect, 1)

        self.screen.blit(grid_surface, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

        # --- Render Ghost Block ---
        if not self.game_over:
            ghost_block = self.current_block.copy()
            while self._is_valid_position(ghost_block, offset_y=1):
                ghost_block["y"] += 1
            self._render_block(ghost_block, alpha=60)
            
            # --- Render Current Block ---
            self._render_block(self.current_block)

        # --- Render Line Clear Animation ---
        time_delta = self.clock.get_time() / 1000.0
        for anim in self.line_clear_animation[:]:
            anim[1] -= time_delta
            if anim[1] <= 0:
                self.line_clear_animation.remove(anim)
            else:
                y = self.GRID_Y_OFFSET + anim[0] * self.GRID_CELL_SIZE
                width = self.GRID_WIDTH * self.GRID_CELL_SIZE
                flash_rect = pygame.Rect(self.GRID_X_OFFSET, y, width, self.GRID_CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_WHITE, flash_rect)

        # --- Render UI ---
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 20))
        
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 20, 20))
        
        # --- Render Next Block ---
        next_text = self.font_small.render("NEXT:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.GRID_X_OFFSET + self.GRID_WIDTH * self.GRID_CELL_SIZE + 20, self.GRID_Y_OFFSET))
        
        next_block_render = self.next_block.copy()
        next_block_render["x"] = (self.GRID_WIDTH + 1.5)
        next_block_render["y"] = self.GRID_HEIGHT / 4
        self._render_block(next_block_render)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.lines_cleared >= self.WIN_CONDITION_LINES else "GAME OVER"
            end_text = self.font_main.render(status_text, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
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

# Example usage for visualization and testing
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows' as appropriate
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tetris-like Gym Environment")
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0)

    # Human play loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # Create action from keyboard state
        keys = pygame.key.get_pressed()
        move_action = 0 # no-op
        if keys[pygame.K_UP]: move_action = 1
        elif keys[pygame.K_DOWN]: move_action = 2
        elif keys[pygame.K_LEFT]: move_action = 3
        elif keys[pygame.K_RIGHT]: move_action = 4
        
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([move_action, space_action, shift_action])

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Lines: {info['lines_cleared']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
    
    env.close()