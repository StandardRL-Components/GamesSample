
# Generated: 2025-08-28T05:32:04.725921
# Source Brief: brief_05604.md
# Brief Index: 5604

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a tile, then move to an adjacent tile and press space again to swap."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Swap adjacent tiles to create matches of 3 or more. Reach 500 points before you run out of 20 moves!"
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
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_SIZE = (8, 8)  # 8x8 grid
        self.NUM_TILE_TYPES = 6
        self.TILE_SIZE = 40
        self.GRID_LINE_WIDTH = 2
        self.BOARD_OFFSET = (
            (640 - self.GRID_SIZE[0] * self.TILE_SIZE) // 2,
            (400 - self.GRID_SIZE[1] * self.TILE_SIZE) // 2 + 20,
        )
        self.WIN_SCORE = 500
        self.MAX_MOVES = 20

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SCORE = (255, 215, 0)
        self.COLOR_MOVES = (173, 216, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SELECT = (100, 255, 100)
        self.TILE_COLORS = [
            (255, 80, 80),   # Red
            (80, 200, 80),   # Green
            (80, 80, 255),   # Blue
            (255, 255, 80),  # Yellow
            (200, 80, 200),  # Purple
            (255, 160, 80),  # Orange
        ]
        
        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.win = False
        self.cursor_pos = (0, 0)
        self.selected_tile = None
        self.last_swap_reward = 0
        self.animations = []

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_valid_grid()
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        self.win = False
        self.cursor_pos = (self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2)
        self.selected_tile = None
        self.animations = [] # Clear any animations
        self.last_swap_reward = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Since auto_advance is False, each step is a discrete turn.
        # Animations will be represented as a sequence of states over multiple steps.
        # This implementation simplifies by resolving swaps and matches instantly for the turn-based model.
        
        reward = 0
        terminated = False
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_press = action[1] == 1  # Boolean
        
        # --- Handle Input and State Changes ---
        if not self.game_over:
            # 1. Handle cursor movement
            if movement == 1: # Up
                self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
            elif movement == 2: # Down
                self.cursor_pos = (self.cursor_pos[0], min(self.GRID_SIZE[1] - 1, self.cursor_pos[1] + 1))
            elif movement == 3: # Left
                self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
            elif movement == 4: # Right
                self.cursor_pos = (min(self.GRID_SIZE[0] - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])

            # 2. Handle selection and swapping
            if space_press:
                if self.selected_tile is None:
                    self.selected_tile = self.cursor_pos
                    # Sound effect: select_tile_1
                else:
                    # Attempt a swap
                    x1, y1 = self.selected_tile
                    x2, y2 = self.cursor_pos
                    
                    # Check for adjacency
                    if abs(x1 - x2) + abs(y1 - y2) == 1:
                        self.moves_left -= 1
                        
                        # Perform swap
                        self._swap_tiles(self.selected_tile, self.cursor_pos)
                        
                        # Check for matches and process them
                        total_reward, chain_count = self._process_all_matches()

                        if total_reward > 0:
                            # Successful swap
                            reward += total_reward
                            # Sound effect: match_success
                        else:
                            # Failed swap, swap back
                            self._swap_tiles(self.selected_tile, self.cursor_pos)
                            reward += -0.1 # Penalty for invalid move
                            # Sound effect: swap_fail
                        
                        self.selected_tile = None
                    else:
                        # Not adjacent, so just re-select
                        self.selected_tile = self.cursor_pos
                        # Sound effect: select_tile_2
            
        # --- Check for Termination ---
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win = True
            reward += 100 # Goal-oriented reward
            terminated = True
        
        if self.moves_left <= 0 and not self.win:
            self.game_over = True
            self.win = False
            reward += -50 # Penalty for losing
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.moves_left, # Using moves_left as a more relevant metric than raw steps
            "cursor": self.cursor_pos,
            "selected": self.selected_tile,
            "is_game_over": self.game_over,
        }

    # --- Grid and Match Logic ---

    def _generate_valid_grid(self):
        """Generates a grid and ensures it has at least one possible move."""
        while True:
            self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=self.GRID_SIZE)
            # Ensure no matches on start
            while self._find_all_matches():
                self.grid = self.np_random.integers(0, self.NUM_TILE_TYPES, size=self.GRID_SIZE)
            
            # Ensure at least one move is possible
            if self._has_possible_moves():
                break

    def _has_possible_moves(self):
        """Checks the entire grid for any valid move."""
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                # Check swap right
                if x < self.GRID_SIZE[0] - 1:
                    self._swap_tiles((x, y), (x + 1, y))
                    if self._find_all_matches():
                        self._swap_tiles((x, y), (x + 1, y)) # Swap back
                        return True
                    self._swap_tiles((x, y), (x + 1, y)) # Swap back
                # Check swap down
                if y < self.GRID_SIZE[1] - 1:
                    self._swap_tiles((x, y), (x, y + 1))
                    if self._find_all_matches():
                        self._swap_tiles((x, y), (x, y + 1)) # Swap back
                        return True
                    self._swap_tiles((x, y), (x, y + 1)) # Swap back
        return False

    def _swap_tiles(self, pos1, pos2):
        """Swaps two tiles in the grid data."""
        x1, y1 = pos1
        x2, y2 = pos2
        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]

    def _find_all_matches(self):
        """Finds all horizontal and vertical matches of 3 or more."""
        matched_tiles = set()
        # Horizontal matches
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0] - 2):
                if self.grid[y, x] == self.grid[y, x+1] == self.grid[y, x+2] and self.grid[y,x] != -1:
                    match = {(x, y), (x+1, y), (x+2, y)}
                    # Extend match
                    i = x + 3
                    while i < self.GRID_SIZE[0] and self.grid[y, i] == self.grid[y, x]:
                        match.add((i, y))
                        i += 1
                    matched_tiles.update(match)

        # Vertical matches
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1] - 2):
                if self.grid[y, x] == self.grid[y+1, x] == self.grid[y+2, x] and self.grid[y,x] != -1:
                    match = {(x, y), (x, y+1), (x, y+2)}
                    # Extend match
                    i = y + 3
                    while i < self.GRID_SIZE[1] and self.grid[i, x] == self.grid[y, x]:
                        match.add((x, i))
                        i += 1
                    matched_tiles.update(match)
        return matched_tiles

    def _process_all_matches(self):
        """Continuously finds matches, removes tiles, and refills grid until no matches are left."""
        total_reward = 0
        chain_count = 0
        while True:
            matches = self._find_all_matches()
            if not matches:
                break
            
            chain_count += 1
            num_matched = len(matches)
            
            # Calculate score and reward
            self.score += num_matched * chain_count
            reward = num_matched
            if num_matched >= 5:
                reward += 10 # Bonus for large cluster
            total_reward += reward

            # Mark tiles for removal
            for x, y in matches:
                self.grid[y, x] = -1

            # Apply gravity
            self._apply_gravity()
            
            # Refill grid
            self._refill_grid()
        
        return total_reward, chain_count

    def _apply_gravity(self):
        """Shifts tiles down to fill empty spaces."""
        for x in range(self.GRID_SIZE[0]):
            empty_spaces = 0
            for y in range(self.GRID_SIZE[1] - 1, -1, -1):
                if self.grid[y, x] == -1:
                    empty_spaces += 1
                elif empty_spaces > 0:
                    self.grid[y + empty_spaces, x] = self.grid[y, x]
                    self.grid[y, x] = -1

    def _refill_grid(self):
        """Fills empty spaces at the top with new random tiles."""
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                if self.grid[y, x] == -1:
                    self.grid[y, x] = self.np_random.integers(0, self.NUM_TILE_TYPES)

    # --- Rendering Methods ---

    def _render_game(self):
        """Renders the grid and tiles."""
        grid_width = self.GRID_SIZE[0] * self.TILE_SIZE
        grid_height = self.GRID_SIZE[1] * self.TILE_SIZE
        
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID, (*self.BOARD_OFFSET, grid_width, grid_height))
        
        # Draw tiles
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                tile_type = self.grid[y, x]
                if tile_type != -1:
                    color = self.TILE_COLORS[tile_type]
                    rect = pygame.Rect(
                        self.BOARD_OFFSET[0] + x * self.TILE_SIZE,
                        self.BOARD_OFFSET[1] + y * self.TILE_SIZE,
                        self.TILE_SIZE,
                        self.TILE_SIZE
                    )
                    # Use a slightly smaller rect for a visual gap
                    inner_rect = rect.inflate(-self.GRID_LINE_WIDTH*2, -self.GRID_LINE_WIDTH*2)
                    pygame.draw.rect(self.screen, color, inner_rect, border_radius=5)
    
    def _render_ui(self):
        """Renders the score, moves, cursors, and game over messages."""
        # Draw Score
        score_surf = self.font_main.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        self.screen.blit(score_surf, (20, 10))

        # Draw Moves Left
        moves_surf = self.font_main.render(f"Moves: {self.moves_left}", True, self.COLOR_MOVES)
        moves_rect = moves_surf.get_rect(topright=(620, 10))
        self.screen.blit(moves_surf, moves_rect)
        
        # Draw selection and cursor
        tile_center_offset = self.TILE_SIZE // 2
        
        # Primary selection
        if self.selected_tile:
            x, y = self.selected_tile
            center_x = self.BOARD_OFFSET[0] + x * self.TILE_SIZE + tile_center_offset
            center_y = self.BOARD_OFFSET[1] + y * self.TILE_SIZE + tile_center_offset
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, tile_center_offset - 4, self.COLOR_SELECT)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, tile_center_offset - 5, self.COLOR_SELECT)

        # Current cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.BOARD_OFFSET[0] + cx * self.TILE_SIZE,
            self.BOARD_OFFSET[1] + cy * self.TILE_SIZE,
            self.TILE_SIZE, self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            message = "You Win!" if self.win else "Game Over"
            color = self.COLOR_SCORE if self.win else self.COLOR_TEXT
            
            text_surf = self.font_large.render(message, True, color)
            text_rect = text_surf.get_rect(center=(320, 180))
            
            final_score_surf = self.font_main.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(320, 240))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(text_surf, text_rect)
            self.screen.blit(final_score_surf, final_score_rect)
            
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
    obs, info = env.reset()
    
    # --- Manual Play Loop ---
    # This loop allows a human to play the game.
    pygame.display.set_caption("Match-3 Game")
    screen = pygame.display.set_mode((640, 400))
    running = True
    terminated = False

    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and not terminated:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                
                # For turn-based, we step on each key press
                obs, reward, terminated, truncated, info = env.step(action)
                
                if reward != 0:
                    print(f"Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['steps']}")
                if terminated:
                    print("Game Over!")

            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()
                terminated = False

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
    env.close()