
# Generated: 2025-08-28T03:39:53.449357
# Source Brief: brief_04992.md
# Brief Index: 4992

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = "Controls: Use arrow keys (↑↓←→) to push all blocks."

    # Short, user-facing description of the game
    game_description = (
        "Strategically push colored blocks to fill the grid and achieve a target fill percentage in a limited number of moves."
    )

    # Frames auto-advance for smooth animation.
    auto_advance = True

    # --- Game Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_W, GRID_H = 12, 8
    CELL_SIZE = 40
    GRID_X_OFFSET = (WIDTH - GRID_W * CELL_SIZE) // 2
    GRID_Y_OFFSET = (HEIGHT - GRID_H * CELL_SIZE) // 2
    MOVE_LIMIT = 50
    WIN_PERCENTAGE = 80.0
    ANIMATION_FRAMES = 8
    INITIAL_FILL_RATIO = 0.25

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_BG = (43, 48, 61)
    COLOR_GRID_LINES = (62, 68, 81)
    COLOR_TEXT = (220, 220, 220)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    COLOR_WIN = (138, 255, 138)
    COLOR_LOSE = (255, 100, 100)
    BLOCK_COLORS = [
        (255, 107, 107),  # Red
        (130, 170, 255),  # Blue
        (255, 214, 107),  # Yellow
        (102, 221, 170),  # Green
        (229, 153, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables
        self.grid = None
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_state = False
        self.last_fill_count = 0
        
        # Animation state
        self.is_animating = False
        self.animation_timer = 0
        self.animation_data = []

        # RNG
        self.rng = np.random.default_rng()
        
        # Initialize state
        self.reset()
        
        # Run self-check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.moves_left = self.MOVE_LIMIT
        self._place_initial_blocks()
        self.last_fill_count = self._get_filled_count()

        # Reset animation state
        self.is_animating = False
        self.animation_timer = 0
        self.animation_data = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = self.game_over

        if self.is_animating:
            self.animation_timer -= 1
            if self.animation_timer <= 0:
                self.is_animating = False
                self.animation_data = []
        elif not terminated:
            if movement > 0:  # A push action was requested
                move_happened = self._handle_push(movement)
                if move_happened:
                    self.moves_left -= 1
                    
                    # Incremental reward
                    new_fill_count = self._get_filled_count()
                    reward += (new_fill_count - self.last_fill_count) * 0.1
                    self.last_fill_count = new_fill_count
                    
                    # Check for termination and apply terminal rewards
                    fill_pct = self._get_fill_percentage()
                    is_win = fill_pct >= self.WIN_PERCENTAGE
                    is_loss = self.moves_left <= 0 and not is_win

                    if is_win:
                        terminated = True
                        self.win_state = True
                        reward += 100
                        # Special bonus for exactly 80%
                        if abs(fill_pct - self.WIN_PERCENTAGE) < 0.01:
                            reward += 5
                    elif is_loss:
                        terminated = True
                        reward -= 100
                    
                    self.game_over = terminated
        
        self.score += reward
        self.steps += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info(),
        )

    def _handle_push(self, direction):
        # 1=up, 2=down, 3=left, 4=right
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[direction]

        # Determine iteration order based on push direction
        x_range = reversed(range(self.GRID_W)) if dx > 0 else range(self.GRID_W)
        y_range = reversed(range(self.GRID_H)) if dy > 0 else range(self.GRID_H)

        new_grid = self.grid.copy()
        self.animation_data = []
        moved_blocks = False

        # Iterate through the grid to find new block positions
        for y_start in y_range:
            for x_start in x_range:
                if new_grid[y_start, x_start] == 0:
                    continue

                x, y = x_start, y_start
                # Find the furthest empty spot
                while 0 <= x + dx < self.GRID_W and 0 <= y + dy < self.GRID_H and new_grid[y + dy, x + dx] == 0:
                    x += dx
                    y += dy
                
                if (x, y) != (x_start, y_start):
                    moved_blocks = True
                    color_id = new_grid[y_start, x_start]
                    
                    # Store animation data
                    start_pos = (x_start, y_start)
                    end_pos = (x, y)
                    self.animation_data.append((start_pos, end_pos, color_id))
                    
                    # Update the grid state
                    new_grid[y, x] = color_id
                    new_grid[y_start, x_start] = 0

        if moved_blocks:
            self.grid = new_grid
            self.is_animating = True
            self.animation_timer = self.ANIMATION_FRAMES
            # Sound effect placeholder: # sfx_block_slide.play()
        
        return moved_blocks

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
            "moves_left": self.moves_left,
            "fill_percentage": self._get_fill_percentage(),
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_W * self.CELL_SIZE, self.GRID_H * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw grid lines
        for i in range(self.GRID_W + 1):
            x = self.GRID_X_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_H * self.CELL_SIZE))
        for i in range(self.GRID_H + 1):
            y = self.GRID_Y_OFFSET + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_W * self.CELL_SIZE, y))

        # Draw blocks
        if self.is_animating:
            self._render_animated_blocks()
        else:
            self._render_static_blocks()

    def _render_static_blocks(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                color_id = self.grid[y, x]
                if color_id > 0:
                    self._draw_block(x, y, color_id)

    def _render_animated_blocks(self):
        interp_t = 1.0 - (self.animation_timer / self.ANIMATION_FRAMES)
        
        # Create a set of blocks that are currently moving
        moving_blocks_coords = set(d[0] for d in self.animation_data)
        
        # Draw static blocks first
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if self.grid[y,x] > 0 and (x,y) not in [d[1] for d in self.animation_data]:
                     self._draw_block(x, y, self.grid[y,x])

        # Draw animated blocks
        for start_pos, end_pos, color_id in self.animation_data:
            start_px = self._grid_to_pixel(start_pos[0], start_pos[1])
            end_px = self._grid_to_pixel(end_pos[0], end_pos[1])
            
            # Linear interpolation (lerp)
            current_px = (
                start_px[0] + (end_px[0] - start_px[0]) * interp_t,
                start_px[1] + (end_px[1] - start_px[1]) * interp_t,
            )
            
            block_rect = pygame.Rect(current_px[0], current_px[1], self.CELL_SIZE, self.CELL_SIZE)
            self._draw_block_pixel(block_rect, color_id)

    def _draw_block(self, grid_x, grid_y, color_id):
        px, py = self._grid_to_pixel(grid_x, grid_y)
        block_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        self._draw_block_pixel(block_rect, color_id)

    def _draw_block_pixel(self, rect, color_id):
        color = self.BLOCK_COLORS[(color_id - 1) % len(self.BLOCK_COLORS)]
        
        # Draw main block with a 3D effect
        pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
        
        # Draw highlight and shadow
        highlight_color = tuple(min(255, c + 40) for c in color)
        shadow_color = tuple(max(0, c - 40) for c in color)
        
        # Top and left highlight
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
        # Bottom and right shadow
        pygame.draw.line(self.screen, shadow_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow_color, rect.topright, rect.bottomright, 2)

    def _render_ui(self):
        fill_pct = self._get_fill_percentage()
        
        # Render Fill Percentage
        fill_text = f"Fill: {fill_pct:.1f}%"
        self._draw_text(fill_text, (self.WIDTH // 2, 20), self.font_medium, self.COLOR_TEXT)

        # Render Moves Left
        moves_text = f"Moves: {self.moves_left}"
        self._draw_text(moves_text, (self.WIDTH // 2, 50), self.font_medium, self.COLOR_TEXT)

        # Render Game Over message
        if self.game_over and not self.is_animating:
            if self.win_state:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
                # Sound effect placeholder: # sfx_win.play()
            else:
                msg = "OUT OF MOVES"
                color = self.COLOR_LOSE
                # Sound effect placeholder: # sfx_lose.play()
            
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2), self.font_large, color)
    
    def _draw_text(self, text, pos, font, color):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=pos)
        
        # Shadow
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        shadow_rect = shadow_surf.get_rect(center=(pos[0]+2, pos[1]+2))
        self.screen.blit(shadow_surf, shadow_rect)
        
        self.screen.blit(text_surf, text_rect)

    def _place_initial_blocks(self):
        self.grid = np.zeros((self.GRID_H, self.GRID_W), dtype=int)
        total_cells = self.GRID_W * self.GRID_H
        num_blocks_to_place = int(total_cells * self.INITIAL_FILL_RATIO)
        
        available_coords = [(y, x) for y in range(self.GRID_H) for x in range(self.GRID_W)]
        chosen_indices = self.rng.choice(len(available_coords), size=num_blocks_to_place, replace=False)
        
        for index in chosen_indices:
            y, x = available_coords[index]
            self.grid[y, x] = self.rng.integers(1, len(self.BLOCK_COLORS) + 1)

    def _get_filled_count(self):
        return np.count_nonzero(self.grid)

    def _get_fill_percentage(self):
        total_cells = self.GRID_W * self.GRID_H
        if total_cells == 0:
            return 0.0
        return (self._get_filled_count() / total_cells) * 100

    def _grid_to_pixel(self, grid_x, grid_y):
        return (
            self.GRID_X_OFFSET + grid_x * self.CELL_SIZE,
            self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE,
        )

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

# Example of how to run the environment
if __name__ == '__main__':
    # For human play
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Prevents window from opening
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # To display the game, you would need a Pygame window.
    # This example just runs the logic.
    print("Environment created. Initial state:")
    print(f"Score: {info['score']}, Moves Left: {info['moves_left']}")

    # Simulate a few random steps
    for i in range(200):
        action = env.action_space.sample()
        # To make it more interesting, prioritize movement actions
        if random.random() < 0.2:
            action[0] = random.randint(1, 4)
        else:
            action[0] = 0 # no-op
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step {i}: Action {action}, Reward: {reward:.2f}, Info: {info}")

        if terminated:
            print("Episode finished!")
            print(f"Final Score: {info['score']}, Moves Left: {info['moves_left']}")
            break
            
    env.close()