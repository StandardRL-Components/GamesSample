
# Generated: 2025-08-28T04:53:42.992126
# Source Brief: brief_02445.md
# Brief Index: 2445

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the selector. Press space to clear a group of same-colored blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Connect and clear groups of 2 or more adjacent blocks to score points. You have 50 moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 6
        self.NUM_COLORS = 5
        self.MAX_MOVES = 50
        self.MIN_CLEAR_SIZE = 2
        
        # --- Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)

        # --- Visuals ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (50, 56, 72)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        self.BLOCK_COLORS = [
            (239, 71, 111),  # Red
            (255, 209, 102), # Yellow
            (6, 214, 160),   # Green
            (17, 138, 178),  # Blue
            (111, 94, 214),  # Purple
        ]
        
        self.BLOCK_SIZE = 50
        self.GRID_WIDTH = self.GRID_HEIGHT = self.GRID_SIZE * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2 + 20

        # --- Game State ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.last_cleared_info = {} # For particle effects
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # For internal testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.last_cleared_info = {}
        
        # Ensure there's at least one valid move on reset
        while not self._has_valid_moves():
             self.grid = self.np_random.integers(0, self.NUM_COLORS, size=(self.GRID_SIZE, self.GRID_SIZE))

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement, space_press, _ = action
        reward = 0
        
        # Clear previous step's particle effects
        self.last_cleared_info = {}
        
        # --- Handle Input ---
        # Movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        # Clamp cursor position
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
        
        # Action: Space press
        if space_press == 1:
            component = self._find_connected_blocks(self.cursor_pos[0], self.cursor_pos[1])
            
            if len(component) >= self.MIN_CLEAR_SIZE:
                # Store info for particle effects
                color_id = self.grid[self.cursor_pos[0], self.cursor_pos[1]]
                self.last_cleared_info = {'positions': component, 'color': self.BLOCK_COLORS[color_id]}

                # Calculate reward and update score
                num_cleared = len(component)
                reward = num_cleared # +1 per block
                self.score += reward
                
                # Decrement moves
                self.moves_left -= 1
                
                # Clear blocks
                cleared_mask = np.zeros_like(self.grid, dtype=bool)
                for x, y in component:
                    cleared_mask[x, y] = True

                # Apply gravity and refill
                for x in range(self.GRID_SIZE):
                    col = self.grid[x, :]
                    survivors = col[~cleared_mask[x, :]]
                    num_new_blocks = self.GRID_SIZE - len(survivors)
                    new_blocks = self.np_random.integers(0, self.NUM_COLORS, size=num_new_blocks)
                    self.grid[x, :] = np.concatenate((new_blocks, survivors))

                # # Sound effect placeholder
                # sfx: block_clear.wav

        # --- Check Termination ---
        terminated = self.moves_left <= 0
        if terminated:
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _find_connected_blocks(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_SIZE and 0 <= start_y < self.GRID_SIZE):
            return []

        target_color = self.grid[start_x, start_y]
        q = [(start_x, start_y)]
        visited = set(q)
        component = []

        while q:
            x, y = q.pop(0)
            component.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_SIZE and 0 <= ny < self.GRID_SIZE and
                        (nx, ny) not in visited and self.grid[nx, ny] == target_color):
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return component
    
    def _has_valid_moves(self):
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if len(self._find_connected_blocks(x,y)) >= self.MIN_CLEAR_SIZE:
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
            "moves_left": self.moves_left,
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.BLOCK_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.BLOCK_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.BLOCK_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.BLOCK_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 2)
        
        # Find potential clear component for highlighting
        highlight_component = self._find_connected_blocks(self.cursor_pos[0], self.cursor_pos[1])
        if len(highlight_component) < self.MIN_CLEAR_SIZE:
            highlight_component = []

        # Draw blocks and highlights
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + x * self.BLOCK_SIZE,
                    self.GRID_OFFSET_Y + y * self.BLOCK_SIZE,
                    self.BLOCK_SIZE, self.BLOCK_SIZE
                )
                
                color_id = self.grid[x, y]
                block_color = self.BLOCK_COLORS[color_id]
                
                # Highlight effect
                if (x, y) in highlight_component:
                    highlight_rect = rect.inflate(8, 8)
                    highlight_color = tuple(min(255, c + 50) for c in block_color)
                    pygame.draw.rect(self.screen, highlight_color, highlight_rect, border_radius=10)

                # Draw main block
                pygame.draw.rect(self.screen, block_color, rect, border_radius=6)
                
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + self.cursor_pos[0] * self.BLOCK_SIZE,
            self.GRID_OFFSET_Y + self.cursor_pos[1] * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=4, border_radius=8)

        # Draw particle effects for one frame
        if 'positions' in self.last_cleared_info:
            # # Sound effect placeholder
            # sfx: particles.wav
            positions = self.last_cleared_info['positions']
            color = self.last_cleared_info['color']
            for x, y in positions:
                center_x = self.GRID_OFFSET_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                center_y = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
                for _ in range(5): # 5 particles per block
                    offset_x = self.np_random.uniform(-1, 1) * self.BLOCK_SIZE * 0.6
                    offset_y = self.np_random.uniform(-1, 1) * self.BLOCK_SIZE * 0.6
                    radius = self.np_random.integers(2, 5)
                    pygame.gfxdraw.filled_circle(
                        self.screen,
                        int(center_x + offset_x),
                        int(center_y + offset_y),
                        radius,
                        color
                    )

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Moves display
        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, (255, 80, 80))
            game_over_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(game_over_text, game_over_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)


    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Beginning implementation validation...")
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

    def close(self):
        pygame.quit()

# This part is for standalone testing and visualization
if __name__ == "__main__":
    env = GameEnv()
    env.validate_implementation()
    
    # To run with human input:
    obs, info = env.reset()
    done = False
    
    # Setup a visible window for human play
    render_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("BlockGrid Puzzle")
    clock = pygame.time.Clock()
    
    action = [0, 0, 0] # no-op, no-space, no-shift
    
    while not done:
        # --- Human Input Handling ---
        # Reset movement for this frame
        action[0] = 0 # No movement unless a key is pressed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action[0] = 1
                elif event.key == pygame.K_DOWN: action[0] = 2
                elif event.key == pygame.K_LEFT: action[0] = 3
                elif event.key == pygame.K_RIGHT: action[0] = 4
                elif event.key == pygame.K_SPACE: action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    action = [0, 0, 0]
                    continue

        # If a key was pressed, step the environment
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Reset button presses for next frame
            action[1] = 0
            action[2] = 0
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        render_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for human play
        
    print(f"Game Over! Final Score: {info['score']}")
    
    # Keep window open for a bit after game over
    end_time = pygame.time.get_ticks() + 3000
    while pygame.time.get_ticks() < end_time:
         for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
         clock.tick(30)

    env.close()