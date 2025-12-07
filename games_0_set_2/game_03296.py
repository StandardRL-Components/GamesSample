import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque
import os
import os
import pygame



class GameEnv(gym.Env):
    """
    A tile-matching puzzle game where the player selects tiles on a grid to find
    and clear groups of 3 or more adjacent, same-colored tiles. The goal is to
    reach a target score within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to select a tile and attempt a match."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match adjacent tiles of the same color in a grid to reach a target score before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Constants ---
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    TILE_SIZE = 40
    GRID_LINE_WIDTH = 2
    SELECTOR_WIDTH = 4
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_SELECTOR = (255, 255, 0)
    TILE_COLORS = [
        (255, 87, 87),    # Red
        (87, 255, 87),    # Green
        (87, 87, 255),    # Blue
        (255, 255, 87),   # Yellow
        (255, 87, 255),   # Magenta
        (87, 255, 255),   # Cyan
    ]
    
    # --- Gameplay Constants ---
    WIN_SCORE = 1000
    MAX_MOVES = 25
    MIN_MATCH_SIZE = 3
    MAX_EPISODE_STEPS = 500

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.render_mode = render_mode
        self.screen_width = 640
        self.screen_height = 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Calculate grid position to center it
        self.grid_pixel_width = self.GRID_WIDTH * self.TILE_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.TILE_SIZE
        self.grid_top_left_x = (self.screen_width - self.grid_pixel_width) // 2
        self.grid_top_left_y = (self.screen_height - self.grid_pixel_height) // 2

        # Initialize state variables
        self.grid = None
        self.selector_pos = None
        self.score = 0
        self.moves_left = 0
        self.steps = 0
        self.game_over = False
        self.last_match_info = {}

        # An initial reset is required to initialize the grid and other state
        # variables before validation or rendering can occur.
        self.reset()

        # Validate implementation after setup
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.selector_pos = np.array([self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2])
        self.last_match_info = {}
        
        self._initialize_grid()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1  # Boolean
        
        self.steps += 1
        reward = 0
        
        # 1. Handle Selector Movement
        if movement == 1:  # Up
            self.selector_pos[1] -= 1
        elif movement == 2:  # Down
            self.selector_pos[1] += 1
        elif movement == 3:  # Left
            self.selector_pos[0] -= 1
        elif movement == 4:  # Right
            self.selector_pos[0] += 1
        
        # Clamp selector position to grid boundaries
        self.selector_pos[0] = np.clip(self.selector_pos[0], 0, self.GRID_WIDTH - 1)
        self.selector_pos[1] = np.clip(self.selector_pos[1], 0, self.GRID_HEIGHT - 1)
        
        # 2. Handle Tile Selection
        if space_pressed:
            self.moves_left -= 1
            # sfx_select_tile
            
            x, y = self.selector_pos
            matched_tiles = self._find_matches(x, y)
            
            if len(matched_tiles) >= self.MIN_MATCH_SIZE:
                # sfx_match_success
                # Score is squared to reward larger matches
                score_gain = len(matched_tiles) ** 2
                self.score += score_gain
                
                # Reward is linear to number of tiles matched
                reward += len(matched_tiles)
                
                self._remove_and_drop_tiles(matched_tiles)
                self.last_match_info = {
                    "count": len(matched_tiles), 
                    "score": score_gain,
                    "pos": (x,y),
                    "frame": self.steps
                }
            else:
                # sfx_match_fail
                self.last_match_info = {}

        # 3. Check for Termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            # sfx_win_game
            reward = 100  # Win bonus
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            # sfx_lose_game
            reward = -10 # Loss penalty
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _initialize_grid(self):
        """Fills the grid with random tiles, ensuring no initial matches."""
        self.grid = self.np_random.integers(0, len(self.TILE_COLORS), size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        
        # Continuously find and replace matches until the board is clear
        while True:
            found_match = False
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    matches = self._find_matches(x, y)
                    if len(matches) >= self.MIN_MATCH_SIZE:
                        found_match = True
                        for mx, my in matches:
                            # Replace with a new random color, avoiding the current one
                            current_color = self.grid[mx, my]
                            new_color = self.np_random.integers(0, len(self.TILE_COLORS))
                            while new_color == current_color:
                                new_color = self.np_random.integers(0, len(self.TILE_COLORS))
                            self.grid[mx, my] = new_color
            if not found_match:
                break
    
    def _find_matches(self, start_x, start_y):
        """Finds all connected tiles of the same color using Breadth-First Search."""
        if not (0 <= start_x < self.GRID_WIDTH and 0 <= start_y < self.GRID_HEIGHT):
            return []

        target_color = self.grid[start_x, start_y]
        if target_color == -1: # Empty tile
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        matches = []

        while q:
            x, y = q.popleft()
            matches.append((x, y))
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and
                    (nx, ny) not in visited and self.grid[nx, ny] == target_color):
                    visited.add((nx, ny))
                    q.append((nx, ny))
                    
        return matches

    def _remove_and_drop_tiles(self, matched_tiles):
        """Removes matched tiles and drops down the tiles above them."""
        # Mark matched tiles as empty (-1)
        for x, y in matched_tiles:
            self.grid[x, y] = -1
            
        # For each column, drop tiles down
        for x in range(self.GRID_WIDTH):
            write_y = self.GRID_HEIGHT - 1
            for read_y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, read_y] != -1:
                    self.grid[x, write_y], self.grid[x, read_y] = self.grid[x, read_y], self.grid[x, write_y]
                    write_y -= 1
            
            # Fill the empty top spaces with new random tiles
            for y in range(write_y, -1, -1):
                self.grid[x, y] = self.np_random.integers(0, len(self.TILE_COLORS))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_tiles()
        self._render_selector()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        """Draws the static grid lines."""
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.grid_top_left_x + x * self.TILE_SIZE, self.grid_top_left_y)
            end_pos = (self.grid_top_left_x + x * self.TILE_SIZE, self.grid_top_left_y + self.grid_pixel_height)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.grid_top_left_x, self.grid_top_left_y + y * self.TILE_SIZE)
            end_pos = (self.grid_top_left_x + self.grid_pixel_width, self.grid_top_left_y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, self.GRID_LINE_WIDTH)
            
    def _render_tiles(self):
        """Draws the colored tiles on the grid."""
        padding = 4
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_index = self.grid[x, y]
                if color_index != -1:
                    tile_color = self.TILE_COLORS[color_index]
                    rect = pygame.Rect(
                        self.grid_top_left_x + x * self.TILE_SIZE + padding,
                        self.grid_top_left_y + y * self.TILE_SIZE + padding,
                        self.TILE_SIZE - 2 * padding,
                        self.TILE_SIZE - 2 * padding,
                    )
                    pygame.draw.rect(self.screen, tile_color, rect, border_radius=5)
                    
                    # Add a subtle inner highlight for depth
                    highlight_color = tuple(min(255, c + 40) for c in tile_color)
                    pygame.draw.rect(self.screen, highlight_color, rect.inflate(-8, -8), border_radius=4)


    def _render_selector(self):
        """Draws the player's selector, with a pulsing glow effect."""
        sel_x, sel_y = self.selector_pos
        rect = pygame.Rect(
            self.grid_top_left_x + sel_x * self.TILE_SIZE,
            self.grid_top_left_y + sel_y * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        
        # Pulsing alpha for the glow
        alpha = 128 + 127 * math.sin(self.steps * 0.3)
        glow_color = (*self.COLOR_SELECTOR, alpha)
        
        # Create a temporary surface for the glow
        glow_surface = pygame.Surface((self.TILE_SIZE + 10, self.TILE_SIZE + 10), pygame.SRCALPHA)
        pygame.draw.rect(glow_surface, glow_color, glow_surface.get_rect(), border_radius=10)
        self.screen.blit(glow_surface, (rect.x - 5, rect.y - 5), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw the main selector border
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, rect, self.SELECTOR_WIDTH, border_radius=8)

    def _render_ui(self):
        """Renders score, moves, and game over messages."""
        # Score display
        score_text = self.font_medium.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))
        
        # Moves display
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, (255, 255, 255))
        self.screen.blit(moves_text, (self.screen_width - moves_text.get_width() - 20, 10))

        # Match feedback text
        if "score" in self.last_match_info and self.last_match_info["score"] > 0:
            if self.steps == self.last_match_info["frame"]:
                feedback_text = self.font_small.render(f"+{self.last_match_info['score']}", True, self.COLOR_SELECTOR)
                sel_x, sel_y = self.last_match_info["pos"]
                text_pos_x = self.grid_top_left_x + sel_x * self.TILE_SIZE
                text_pos_y = self.grid_top_left_y + sel_y * self.TILE_SIZE - 20
                self.screen.blit(feedback_text, (text_pos_x, text_pos_y))
        
        # Game Over / Win screen
        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
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
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set SDL to dummy to run headless
    # import os # Already imported at top level
    # os.environ["SDL_VIDEODRIVER"] = "dummy" # This is now done in __init__

    env = GameEnv(render_mode="rgb_array")
    
    # --- To play interactively (requires a window) ---
    # To run this part, comment out the os.environ line above
    # and change render_mode to 'human' if it were supported.
    # Since it's not, we'll simulate it.
    
    # Re-init with a window
    # if "SDL_VIDEODRIVER" in os.environ:
    #     del os.environ["SDL_VIDEODRIVER"]
    # pygame.display.init()
    # pygame.display.set_caption("Tile Matcher")
    # human_screen = pygame.display.set_mode((640, 400))
    # env = GameEnv()

    # obs, info = env.reset()
    # done = False
    # clock = pygame.time.Clock()

    # print(env.user_guide)

    # while not done:
    #     movement, space, shift = 0, 0, 0
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_UP: movement = 1
    #             elif event.key == pygame.K_DOWN: movement = 2
    #             elif event.key == pygame.K_LEFT: movement = 3
    #             elif event.key == pygame.K_RIGHT: movement = 4
    #             elif event.key == pygame.K_SPACE: space = 1
    #             elif event.key == pygame.K_LSHIFT: shift = 1
        
    #     action = [movement, space, shift]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Render to the human-visible screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     human_screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     if reward != 0:
    #         print(f"Step: {info['steps']}, Score: {info['score']}, Moves: {info['moves_left']}, Reward: {reward}")

    #     if done:
    #         print("Game Over!")
    #         pygame.time.wait(2000)

    # env.close()