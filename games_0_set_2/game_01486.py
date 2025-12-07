
# Generated: 2025-08-27T17:18:19.401187
# Source Brief: brief_01486.md
# Brief Index: 1486

        
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
        "Controls: Use arrow keys to move the cursor. Press space to reveal a tile. Match all pairs to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A memory puzzle game. Find all 8 pairs of numbers on the 4x4 grid before you make 3 mistakes."
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Visuals
        self.COLOR_BG = (44, 62, 80) # Dark Blue/Grey
        self.COLOR_GRID_BG = (35, 49, 64)
        self.COLOR_TILE_HIDDEN = (127, 140, 141) # Grey
        self.COLOR_TILE_REVEALED = (149, 165, 166) # Lighter Grey
        self.COLOR_CURSOR = (241, 196, 15) # Yellow
        self.COLOR_TEXT = (236, 240, 241) # White
        self.COLOR_MATCH = (46, 204, 113) # Green
        self.COLOR_MISMATCH = (231, 76, 60) # Red
        
        self.FONT_LARGE = pygame.font.SysFont("monospace", 48, bold=True)
        self.FONT_MEDIUM = pygame.font.SysFont("monospace", 24, bold=True)
        self.FONT_SMALL = pygame.font.SysFont("monospace", 16)
        
        # Game layout constants
        self.GRID_DIM = 4
        self.TILE_SIZE = 80
        self.GAP_SIZE = 10
        self.GRID_WIDTH = self.GRID_DIM * self.TILE_SIZE + (self.GRID_DIM - 1) * self.GAP_SIZE
        self.GRID_HEIGHT = self.GRID_WIDTH
        self.GRID_X = (self.screen_width - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.screen_height - self.GRID_HEIGHT) // 2
        
        # Max steps
        self.MAX_STEPS = 1000
        self.MAX_INCORRECT_GUESSES = 3
        
        # Initialize state variables
        self.grid = None
        self.revealed_mask = None
        self.matched_mask = None
        self.cursor_pos = None
        self.first_selection = None
        self.incorrect_guesses = None
        self.matches_found = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.prev_space_held = None
        
        # Animation state
        self.animation_timer = 0
        self.animation_type = None
        self.animation_tiles = []
        
        self.reset()
        # self.validate_implementation() # Uncomment to run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        numbers = list(range(1, (self.GRID_DIM * self.GRID_DIM // 2) + 1)) * 2
        self.np_random.shuffle(numbers)
        self.grid = np.array(numbers).reshape((self.GRID_DIM, self.GRID_DIM))
        
        self.revealed_mask = np.full((self.GRID_DIM, self.GRID_DIM), False)
        self.matched_mask = np.full((self.GRID_DIM, self.GRID_DIM), False)
        
        self.cursor_pos = [0, 0]
        self.first_selection = None
        
        self.incorrect_guesses = 0
        self.matches_found = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.prev_space_held = False
        
        self.animation_timer = 0
        self.animation_type = None
        self.animation_tiles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        terminated = False
        self.steps += 1

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Animation State ---
        if self.animation_timer > 0:
            self.animation_timer -= 1
            if self.animation_timer == 0:
                if self.animation_type == 'incorrect':
                    # Hide the mismatched tiles after animation
                    r1, c1 = self.animation_tiles[0]
                    r2, c2 = self.animation_tiles[1]
                    self.revealed_mask[r1, c1] = False
                    self.revealed_mask[r2, c2] = False
                self.animation_type = None
                self.animation_tiles = []
            
            # Pause game logic during animation
            self.prev_space_held = space_held
            return self._get_observation(), 0, False, False, self._get_info()

        # --- Input Handling ---
        # Movement
        if movement == 1: # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1) % self.GRID_DIM
        elif movement == 2: # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_DIM
        elif movement == 3: # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1) % self.GRID_DIM
        elif movement == 4: # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_DIM

        # Action (Space Press)
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            r, c = self.cursor_pos
            
            # Ignore clicks on already matched or revealed tiles
            if self.matched_mask[r, c] or self.revealed_mask[r, c]:
                reward -= 0.1 # Small penalty for redundant action
            else:
                self.revealed_mask[r, c] = True
                
                if self.first_selection is None:
                    # This is the first tile of a pair
                    self.first_selection = (r, c)
                    reward += 1.0 # Reward for exploration
                else:
                    # This is the second tile, check for a match
                    r1, c1 = self.first_selection
                    r2, c2 = r, c
                    
                    if self.grid[r1, c1] == self.grid[r2, c2]:
                        # --- CORRECT MATCH ---
                        # sfx: correct_match.wav
                        reward += 10.0
                        self.score += 10
                        self.matches_found += 1
                        self.matched_mask[r1, c1] = True
                        self.matched_mask[r2, c2] = True
                        self.revealed_mask.fill(False)
                        
                        self.animation_type = 'correct'
                        self.animation_timer = 3 # frames
                        self.animation_tiles = [(r1, c1), (r2, c2)]
                    else:
                        # --- INCORRECT MATCH ---
                        # sfx: incorrect_match.wav
                        reward -= 5.0
                        self.score -= 5
                        self.incorrect_guesses += 1
                        
                        self.animation_type = 'incorrect'
                        self.animation_timer = 5 # frames
                        self.animation_tiles = [(r1, c1), (r2, c2)]

                    self.first_selection = None

        self.prev_space_held = space_held
        
        # --- Termination Check ---
        if self.matches_found == (self.GRID_DIM * self.GRID_DIM // 2):
            reward += 50.0 # Win bonus
            terminated = True
            self.game_over = True
        elif self.incorrect_guesses >= self.MAX_INCORRECT_GUESSES:
            reward -= 50.0 # Loss penalty
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        grid_bg_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_bg_rect, border_radius=5)
        
        # Draw tiles
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                tile_x = self.GRID_X + c * (self.TILE_SIZE + self.GAP_SIZE)
                tile_y = self.GRID_Y + r * (self.TILE_SIZE + self.GAP_SIZE)
                tile_rect = pygame.Rect(tile_x, tile_y, self.TILE_SIZE, self.TILE_SIZE)
                
                if self.matched_mask[r, c]:
                    # Matched tiles are faded out
                    pygame.draw.rect(self.screen, self.COLOR_GRID_BG, tile_rect, border_radius=5)
                elif self.revealed_mask[r, c]:
                    # Revealed tiles show their number
                    pygame.draw.rect(self.screen, self.COLOR_TILE_REVEALED, tile_rect, border_radius=5)
                    num_text = self.FONT_LARGE.render(str(self.grid[r, c]), True, self.COLOR_TEXT)
                    text_rect = num_text.get_rect(center=tile_rect.center)
                    self.screen.blit(num_text, text_rect)
                else:
                    # Hidden tiles
                    pygame.draw.rect(self.screen, self.COLOR_TILE_HIDDEN, tile_rect, border_radius=5)
        
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_x = self.GRID_X + cursor_c * (self.TILE_SIZE + self.GAP_SIZE)
        cursor_y = self.GRID_Y + cursor_r * (self.TILE_SIZE + self.GAP_SIZE)
        cursor_rect = pygame.Rect(cursor_x - 4, cursor_y - 4, self.TILE_SIZE + 8, self.TILE_SIZE + 8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=8)

        # Draw animation overlay
        if self.animation_timer > 0:
            color = self.COLOR_MATCH if self.animation_type == 'correct' else self.COLOR_MISMATCH
            alpha = int(150 * (self.animation_timer / 5.0))
            
            overlay_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
            overlay_surface.fill((*color, alpha))
            
            for r, c in self.animation_tiles:
                tile_x = self.GRID_X + c * (self.TILE_SIZE + self.GAP_SIZE)
                tile_y = self.GRID_Y + r * (self.TILE_SIZE + self.GAP_SIZE)
                self.screen.blit(overlay_surface, (tile_x, tile_y))

    def _render_ui(self):
        # Score
        score_text = self.FONT_MEDIUM.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.screen_height - 40))

        # Matches
        matches_text = self.FONT_MEDIUM.render(f"Matches: {self.matches_found} / {self.GRID_DIM**2 // 2}", True, self.COLOR_TEXT)
        matches_rect = matches_text.get_rect(topright=(self.screen_width - 20, 20))
        self.screen.blit(matches_text, matches_rect)
        
        # Guesses
        guesses_left = max(0, self.MAX_INCORRECT_GUESSES - self.incorrect_guesses)
        guesses_color = self.COLOR_TEXT if guesses_left > 1 else self.COLOR_MISMATCH
        guesses_text = self.FONT_MEDIUM.render(f"Guesses Left: {guesses_left}", True, guesses_color)
        self.screen.blit(guesses_text, (20, 20))
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.matches_found == (self.GRID_DIM**2 // 2):
                msg = "YOU WIN!"
                color = self.COLOR_MATCH
            else:
                msg = "GAME OVER"
                color = self.COLOR_MISMATCH
            
            end_text = self.FONT_LARGE.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "matches_found": self.matches_found,
            "incorrect_guesses": self.incorrect_guesses,
            "cursor_pos": self.cursor_pos,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
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
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False
    }

    # Since auto_advance is False, we need a display window
    # and a main loop that calls step on user input.
    display_screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Memory Grid")
    
    running = True
    while running:
        action = [0, 0, 0] # Default no-op action
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False
        
        # Map held keys to MultiDiscrete action
        if keys_held[pygame.K_UP]:
            action[0] = 1
        elif keys_held[pygame.K_DOWN]:
            action[0] = 2
        elif keys_held[pygame.K_LEFT]:
            action[0] = 3
        elif keys_held[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys_held[pygame.K_SPACE]:
            action[1] = 1
            
        if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print("Game Over!")
            print(f"Final Score: {info['score']}, Total Steps: {info['steps']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

    env.close()