
# Generated: 2025-08-28T04:32:51.124673
# Source Brief: brief_05284.md
# Brief Index: 5284

        
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
        "Controls: ↑/↓ to rotate, Space to drop the shape."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Rotate falling shapes to fill the grid. Clear rows for points. "
        "You have 15 moves to fill the entire grid!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # --- Game Constants ---
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 30
        self.GRID_X_OFFSET = 100
        self.GRID_Y_OFFSET = 50
        self.MAX_MOVES = 15
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (26, 26, 29)  # #1A1A1D
        self.COLOR_GRID = (78, 78, 80)  # #4E4E50
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURRENT_PIECE = (255, 255, 255)
        self.SHAPE_COLORS = [
            (50, 50, 50),      # 0: Empty (not used for drawing)
            (0, 240, 240),     # 1: I (Cyan)
            (240, 240, 0),     # 2: O (Yellow)
            (160, 0, 240),     # 3: T (Purple)
            (0, 0, 240),       # 4: J (Blue)
            (240, 160, 0),     # 5: L (Orange)
            (0, 240, 0),       # 6: S (Green)
            (240, 0, 0),       # 7: Z (Red)
        ]

        # --- Shape Definitions (relative to pivot) ---
        self.SHAPES = {
            1: [[0, -1], [0, 0], [0, 1], [0, 2]],  # I
            2: [[0, 0], [1, 0], [0, 1], [1, 1]],  # O
            3: [[-1, 0], [0, 0], [1, 0], [0, 1]],  # T
            4: [[-1, 1], [-1, 0], [0, 0], [1, 0]], # J
            5: [[-1, 0], [0, 0], [1, 0], [1, 1]],  # L
            6: [[-1, 0], [0, 0], [0, 1], [1, 1]],  # S
            7: [[-1, 1], [0, 1], [0, 0], [1, 0]],  # Z
        }

        # --- Game State (initialized in reset) ---
        self.np_random = None
        self.grid = None
        self.current_shape = None
        self.next_shape = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False

        self.current_shape = self._new_shape()
        self.next_shape = self._new_shape()

        if self._check_collision(self._get_shape_coords(self.current_shape)):
             self.game_over = True
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        reward = 0
        
        # Action: Rotate shape (1=CW, 2=CCW)
        if movement in [1, 2]:
            self._rotate_current_shape(1 if movement == 1 else -1)
        
        # Action: Drop shape (this constitutes a "turn")
        if space_pressed:
            reward = self._place_shape()
            
            # Check termination conditions
            if self.win:
                reward += 100
                self.game_over = True
            elif self.moves_left <= 0:
                self.game_over = True
            
            # Spawn next piece if game is not over
            if not self.game_over:
                self.current_shape = self.next_shape
                self.next_shape = self._new_shape()
                # Check for block-out (game over if new piece spawns in an occupied space)
                if self._check_collision(self._get_shape_coords(self.current_shape)):
                    self.game_over = True

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _place_shape(self):
        self.moves_left -= 1
        
        # Find landing position by checking for collision downwards
        shadow_shape = self.current_shape.copy()
        while not self._check_collision(self._get_shape_coords(shadow_shape)):
            shadow_shape['y'] += 1
        shadow_shape['y'] -= 1
        
        # Stamp shape onto grid
        coords = self._get_shape_coords(shadow_shape)
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.current_shape['id']
        
        # Check for and clear full rows
        rows_cleared = 0
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_HEIGHT - 1
        for row_idx in range(self.GRID_HEIGHT - 1, -1, -1):
            if not np.all(self.grid[row_idx, :] > 0):
                new_grid[new_row_idx, :] = self.grid[row_idx, :]
                new_row_idx -= 1
            else:
                rows_cleared += 1
        self.grid = new_grid
        
        # Calculate reward
        reward = 0
        if rows_cleared > 0:
            # +10 for each completed row
            reward += rows_cleared * 10
            self.score += rows_cleared * 10
            # sfx: line clear
        
        # +1 for each partially filled row
        partial_fill_reward = np.sum(
            (np.sum(self.grid > 0, axis=1) > 0) & 
            (np.sum(self.grid > 0, axis=1) < self.GRID_WIDTH)
        )
        reward += partial_fill_reward
        self.score += partial_fill_reward

        # Check for win condition
        self.win = np.all(self.grid > 0)
        if self.win:
            self.score += 100
            # sfx: win fanfare

        return reward

    def _rotate_current_shape(self, direction):
        original_rotation = self.current_shape['rotation']
        self.current_shape['rotation'] = (self.current_shape['rotation'] + direction) % 4
        
        coords = self._get_shape_coords(self.current_shape)
        if self._check_collision(coords):
            # If collision, try to "wall kick"
            for dx in [-1, 1, -2, 2]:
                self.current_shape['x'] += dx
                if not self._check_collision(self._get_shape_coords(self.current_shape)):
                    # sfx: rotate
                    return
                self.current_shape['x'] -= dx # revert kick
            
            # If all kicks fail, revert rotation
            self.current_shape['rotation'] = original_rotation
        else:
            # sfx: rotate
            pass

    def _new_shape(self):
        shape_id = self.np_random.integers(1, len(self.SHAPES) + 1)
        return {
            'id': shape_id,
            'template': self.SHAPES[shape_id],
            'x': self.GRID_WIDTH // 2,
            'y': 1 if shape_id != 1 else 2, # I-shape needs more space to spawn
            'rotation': 0
        }

    def _get_shape_coords(self, shape):
        coords = []
        for rel_x, rel_y in shape['template']:
            # Apply rotation using matrix multiplication logic
            # 0 deg: (x, y), 90 deg CW: (y, -x), 180 deg: (-x, -y), 270 deg CW: (-y, x)
            if shape['rotation'] == 1:
                rel_x, rel_y = rel_y, -rel_x
            elif shape['rotation'] == 2:
                rel_x, rel_y = -rel_x, -rel_y
            elif shape['rotation'] == 3:
                rel_x, rel_y = -rel_y, rel_x
            
            coords.append((shape['x'] + rel_x, shape['y'] + rel_y))
        return coords

    def _check_collision(self, coords):
        for x, y in coords:
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True # Out of bounds
            if self.grid[y, x] > 0:
                return True # Collides with existing block
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "moves_left": self.moves_left, "steps": self.steps}

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, y))

        # Draw filled cells
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] > 0:
                    color_idx = self.grid[y, x]
                    color = self.SHAPE_COLORS[color_idx]
                    rect = pygame.Rect(self.GRID_X_OFFSET + x * self.CELL_SIZE + 1,
                                       self.GRID_Y_OFFSET + y * self.CELL_SIZE + 1,
                                       self.CELL_SIZE - 1, self.CELL_SIZE - 1)
                    pygame.draw.rect(self.screen, color, rect, border_radius=3)

        if self.game_over:
            return

        # Draw shadow piece for placement feedback
        shadow_shape = self.current_shape.copy()
        while not self._check_collision(self._get_shape_coords(shadow_shape)):
            shadow_shape['y'] += 1
        shadow_shape['y'] -= 1
        shadow_coords = self._get_shape_coords(shadow_shape)
        shadow_color = self.SHAPE_COLORS[self.current_shape['id']]
        for x, y in shadow_coords:
            if 0 <= y < self.GRID_HEIGHT:
                rect = pygame.Rect(self.GRID_X_OFFSET + x * self.CELL_SIZE + 1,
                                   self.GRID_Y_OFFSET + y * self.CELL_SIZE + 1,
                                   self.CELL_SIZE - 1, self.CELL_SIZE - 1)
                s = pygame.Surface((self.CELL_SIZE - 1, self.CELL_SIZE - 1), pygame.SRCALPHA)
                s.fill((*shadow_color, 60))
                self.screen.blit(s, rect.topleft)

        # Draw current piece (bright white)
        current_coords = self._get_shape_coords(self.current_shape)
        for x, y in current_coords:
            rect = pygame.Rect(self.GRID_X_OFFSET + x * self.CELL_SIZE + 1,
                               self.GRID_Y_OFFSET + y * self.CELL_SIZE + 1,
                               self.CELL_SIZE - 1, self.CELL_SIZE - 1)
            pygame.draw.rect(self.screen, self.COLOR_CURRENT_PIECE, rect, border_radius=3)

    def _render_ui(self):
        # Draw main UI text
        moves_text = self.font_small.render(f"Moves Left: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(630, 10))
        self.screen.blit(score_text, score_rect)

        # Draw "Next" piece preview
        next_text = self.font_small.render("Next:", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (450, 50))
        
        preview_shape = self.next_shape.copy()
        preview_shape.update({'x': 0, 'y': 0, 'rotation': 0})
        preview_coords = self._get_shape_coords(preview_shape)
        
        min_x = min(c[0] for c in preview_coords) if preview_coords else 0
        max_x = max(c[0] for c in preview_coords) if preview_coords else 0
        min_y = min(c[1] for c in preview_coords) if preview_coords else 0
        max_y = max(c[1] for c in preview_coords) if preview_coords else 0
        
        shape_width = (max_x - min_x + 1) * self.CELL_SIZE
        shape_height = (max_y - min_y + 1) * self.CELL_SIZE
        
        preview_base_x = 450 + (150 - shape_width) / 2
        preview_base_y = 100 + (100 - shape_height) / 2

        for x, y in preview_coords:
            draw_x = preview_base_x + (x - min_x) * self.CELL_SIZE
            draw_y = preview_base_y + (y - min_y) * self.CELL_SIZE
            rect = pygame.Rect(draw_x, draw_y, self.CELL_SIZE - 1, self.CELL_SIZE - 1)
            pygame.draw.rect(self.screen, self.COLOR_CURRENT_PIECE, rect, border_radius=3)

        # Draw Game Over / Win message overlay
        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(320, 200))
            self.screen.blit(end_text, end_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Set up the display window
    pygame.display.set_caption("Geometric Grid Filler")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()

    print(env.user_guide)
    
    action = [0, 0, 0] # [movement, space, shift]

    while not done:
        # --- Event Handling for human play ---
        action_taken = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                action_taken = True
                if event.key == pygame.K_UP:
                    action[0] = 1 # Rotate CW
                elif event.key == pygame.K_DOWN:
                    action[0] = 2 # Rotate CCW
                elif event.key == pygame.K_SPACE:
                    action[1] = 1 # Drop
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                else:
                    action_taken = False # Not a game action

        # --- Step the environment ---
        # Since auto_advance=False, we only step when a meaningful action is taken
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Moves: {info['moves_left']}")
            # Reset non-persistent actions after they are processed
            action = [0, 0, 0]

        # --- Rendering ---
        # The environment's observation is already a rendered frame
        frame = env._get_observation()
        # The observation is (H, W, C) but pygame wants (W, H, C)
        # surfarray.make_surface expects a (W, H) surface, so we transpose
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit frame rate for human play

    env.close()
    pygame.quit()