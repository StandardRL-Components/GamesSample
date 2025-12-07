
# Generated: 2025-08-27T12:27:16.675269
# Source Brief: brief_00048.md
# Brief Index: 48

        
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
    """
    A memory puzzle game where the player reveals tiles to find matching pairs.
    The goal is to clear the board within a limited number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to reveal a tile. Match all pairs before you run out of moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist memory puzzle. Reveal tiles to find matching pairs, but watch your move count! A test of memory and strategy."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        """
        Initializes the game environment, including Pygame, spaces, and constants.
        """
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.grid_size = (4, 4)  # 16 tiles, 8 pairs
        self.tile_size = 80
        self.gap_size = 10
        self.grid_origin_x = (self.screen_width - (self.grid_size[1] * (self.tile_size + self.gap_size) - self.gap_size)) // 2
        self.grid_origin_y = (self.screen_height - (self.grid_size[0] * (self.tile_size + self.gap_size) - self.gap_size)) // 2

        # Visuals
        self.color_bg = (30, 30, 40)
        self.color_tile_hidden = (100, 100, 110)
        self.color_tile_revealed = (220, 220, 230)
        self.color_tile_matched = (100, 200, 100)
        self.color_cursor = (100, 150, 255)
        self.color_text = (255, 255, 255)
        self.shape_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
            (255, 100, 255), (100, 255, 255), (255, 150, 50), (150, 50, 255)
        ]
        self.font_main = pygame.font.SysFont("monospace", 24)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Game state variables (initialized in reset)
        self.grid = None
        self.cursor_pos = None
        self.moves_remaining = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.revealed_pair_coords = None
        self.mismatch_to_hide = None

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        """
        Resets the game to its initial state.
        """
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.moves_remaining = 30
        self.revealed_pair_coords = []
        self.mismatch_to_hide = []

        # Create and shuffle tiles
        num_tiles = self.grid_size[0] * self.grid_size[1]
        num_pairs = num_tiles // 2
        tile_ids = list(range(num_pairs)) * 2
        self.np_random.shuffle(tile_ids)
        
        self.grid = [{'id': tile_id, 'state': 'hidden'} for tile_id in tile_ids]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        """
        Advances the game state by one step based on the given action.
        """
        reward = 0  # No default reward/penalty per step, only for specific actions
        
        # First, handle state cleanup from the previous action (hiding a mismatch)
        if self.mismatch_to_hide:
            for r, c in self.mismatch_to_hide:
                idx = r * self.grid_size[1] + c
                if self.grid[idx]['state'] == 'revealed':
                    self.grid[idx]['state'] = 'hidden'
            self.mismatch_to_hide = []
            self.revealed_pair_coords = []

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        action_taken = False

        # Handle cursor movement
        if movement != 0:
            action_taken = True
            if movement == 1: self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.grid_size[0]  # Up
            elif movement == 2: self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.grid_size[0]  # Down
            elif movement == 3: self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.grid_size[1]  # Left
            elif movement == 4: self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.grid_size[1]  # Right

        # Handle reveal action
        if space_pressed:
            r, c = self.cursor_pos
            idx = r * self.grid_size[1] + c
            tile = self.grid[idx]

            # Only reveal a hidden tile if less than 2 tiles are currently revealed
            if tile['state'] == 'hidden' and len(self.revealed_pair_coords) < 2:
                action_taken = True
                tile['state'] = 'revealed'
                self.revealed_pair_coords.append((r, c))
                self.moves_remaining -= 1
                # sfx: tile_flip.wav

                # If two tiles are now revealed, check for a match
                if len(self.revealed_pair_coords) == 2:
                    r1, c1 = self.revealed_pair_coords[0]
                    r2, c2 = self.revealed_pair_coords[1]
                    idx1 = r1 * self.grid_size[1] + c1
                    idx2 = r2 * self.grid_size[1] + c2
                    
                    if self.grid[idx1]['id'] == self.grid[idx2]['id']:
                        # It's a match!
                        # sfx: match_found.wav
                        self.grid[idx1]['state'] = 'matched'
                        self.grid[idx2]['state'] = 'matched'
                        self.revealed_pair_coords = []
                        reward += 1.0
                        self.score += 1.0
                    else:
                        # It's a mismatch
                        # sfx: mismatch.wav
                        reward += -0.1
                        self.score -= 0.1
                        self.mismatch_to_hide = list(self.revealed_pair_coords)

        if action_taken:
            self.steps += 1

        # Check for termination conditions
        all_matched = all(tile['state'] == 'matched' for tile in self.grid)
        if all_matched:
            self.game_over = True
            reward += 10.0  # Terminal win bonus
            self.score += 10.0
        elif self.moves_remaining <= 0:
            self.game_over = True
            # No specific penalty for running out of moves, low score is sufficient
        elif self.steps >= 1000: # Safety termination
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
    
    def _get_observation(self):
        """
        Renders the current game state to a numpy array.
        """
        # Clear screen with background
        self.screen.fill(self.color_bg)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, tiles, and cursor."""
        # Draw tiles
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                idx = r * self.grid_size[1] + c
                tile = self.grid[idx]
                
                rect_x = self.grid_origin_x + c * (self.tile_size + self.gap_size)
                rect_y = self.grid_origin_y + r * (self.tile_size + self.gap_size)
                tile_rect = pygame.Rect(rect_x, rect_y, self.tile_size, self.tile_size)
                
                color_map = {
                    'hidden': self.color_tile_hidden,
                    'revealed': self.color_tile_revealed,
                    'matched': self.color_tile_matched
                }
                color = color_map[tile['state']]
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=8)
                
                if tile['state'] != 'hidden':
                    self._draw_shape(tile['id'], tile_rect)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect_x = self.grid_origin_x + cursor_c * (self.tile_size + self.gap_size) - 4
        cursor_rect_y = self.grid_origin_y + cursor_r * (self.tile_size + self.gap_size) - 4
        cursor_rect = pygame.Rect(cursor_rect_x, cursor_rect_y, self.tile_size + 8, self.tile_size + 8)
        pygame.draw.rect(self.screen, self.color_cursor, cursor_rect, width=4, border_radius=12)

    def _draw_shape(self, shape_id, rect):
        """Draws a unique geometric shape inside a tile."""
        center_x, center_y = rect.center
        size = self.tile_size // 3
        color = self.shape_colors[shape_id % len(self.shape_colors)]
        shape_type = shape_id % 4

        if shape_type == 0:  # Circle
            pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), int(size), color)
            pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), int(size), color)
        elif shape_type == 1:  # Square
            square_rect = pygame.Rect(center_x - size, center_y - size, size * 2, size * 2)
            pygame.draw.rect(self.screen, color, square_rect, border_radius=4)
        elif shape_type == 2:  # Triangle
            points = [
                (center_x, center_y - size), (center_x - size, center_y + size),
                (center_x + size, center_y + size)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
        elif shape_type == 3:  # Diamond
            points = [
                (center_x, center_y - size), (center_x - size, center_y),
                (center_x, center_y + size), (center_x + size, center_y)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], color)

    def _render_ui(self):
        """Renders UI elements like score, moves, and game over messages."""
        # Moves remaining
        moves_text = self.font_main.render(f"Moves: {max(0, self.moves_remaining)}", True, self.color_text)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.color_text)
        score_rect = score_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(score_text, score_rect)

        # Game over message
        if self.game_over:
            all_matched = all(tile['state'] == 'matched' for tile in self.grid)
            msg, color = ("YOU WIN!", self.color_tile_matched) if all_matched else ("GAME OVER", (200, 80, 80))
            
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        """
        Returns a dictionary with auxiliary diagnostic information.
        """
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
            "cursor_pos": list(self.cursor_pos)
        }

    def close(self):
        """
        Cleans up Pygame resources upon closing the environment.
        """
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # Use a window to display the game
    pygame.display.set_caption("Memory Grid")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Map keyboard keys to the MultiDiscrete action space
        movement = 0  # no-op
        space_pressed = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_pressed = 1
                elif event.key == pygame.K_r: # Reset on 'r' key
                    obs, info = env.reset()
                    movement, space_pressed = 0, 0
                
        # Only step if an action was taken
        if movement != 0 or space_pressed != 0:
            action = [movement, space_pressed, 0] # shift is unused
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.1f}, Done: {done}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # If game is over, wait for a moment before closing
        if done:
            pygame.time.wait(2000)

    env.close()