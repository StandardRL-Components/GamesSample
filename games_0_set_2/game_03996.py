
# Generated: 2025-08-28T01:04:48.027959
# Source Brief: brief_03996.md
# Brief Index: 3996

        
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
        "Controls: Use arrow keys to swap the selected pixel. Use Space/Shift to cycle through pixels."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A pixel puzzle game. Rearrange the shuffled pixels on the grid to match the target image before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_DIM = 10
        self.MAX_MOVES = 20
        self.MAX_STEPS = 1000
        self.LEVEL_COMPLEXITY_START = 1
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.FONT_BIG = pygame.font.SysFont("Consolas", 48, bold=True)
        self.FONT_UI = pygame.font.SysFont("Consolas", 20, bold=True)
        self.FONT_SMALL = pygame.font.SysFont("Consolas", 14)

        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_BG = (40, 45, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)
        self.COLOR_SELECT = (255, 255, 255)

        self.PALETTE = [
            (255, 60, 60),    # Red
            (60, 255, 60),    # Green
            (60, 120, 255),   # Blue
            (255, 255, 60),   # Yellow
            (255, 60, 255),   # Magenta
            (60, 255, 255),   # Cyan
            (255, 255, 255),  # White
            (10, 10, 10),     # Black
        ]

        # Rendering layout
        self.GRID_PIXEL_SIZE = 32
        self.GRID_AREA_SIZE = self.GRID_DIM * self.GRID_PIXEL_SIZE
        self.GRID_TOP_LEFT = (
            (self.WIDTH - self.GRID_AREA_SIZE) // 2,
            (self.HEIGHT - self.GRID_AREA_SIZE) // 2
        )
        
        self.TARGET_PIXEL_SIZE = 10
        self.TARGET_AREA_SIZE = self.GRID_DIM * self.TARGET_PIXEL_SIZE
        self.TARGET_TOP_LEFT = (self.WIDTH - self.TARGET_AREA_SIZE - 20, 20)
        
        # Initialize state variables
        self.level_complexity = self.LEVEL_COMPLEXITY_START
        self.np_random = None # Will be initialized in reset
        self.current_grid = None
        self.target_grid = None
        self.selected_idx = 0
        self.moves_left = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.reset()
        
        # self.validate_implementation() # Uncomment for testing

    def _generate_target_grid(self):
        grid = np.full((self.GRID_DIM, self.GRID_DIM), self.np_random.integers(len(self.PALETTE)), dtype=int)
        for _ in range(self.level_complexity):
            color = self.np_random.integers(len(self.PALETTE))
            x, y = self.np_random.integers(self.GRID_DIM, size=2)
            w, h = self.np_random.integers(1, self.GRID_DIM // 2 + 1, size=2)
            grid[y:y+h, x:x+w] = color
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        if options and 'level_complexity' in options:
            self.level_complexity = options['level_complexity']

        self.target_grid = self._generate_target_grid()
        shuffled_pixels = self.target_grid.flatten()
        self.np_random.shuffle(shuffled_pixels)
        self.current_grid = shuffled_pixels.reshape((self.GRID_DIM, self.GRID_DIM))

        self.selected_idx = 0
        self.moves_left = self.MAX_MOVES
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        return self._get_observation(), self._get_info()
    
    def _calculate_match_reward(self, grid):
        """Calculates a score based on how well the grid matches the target."""
        pixel_matches = np.sum(grid == self.target_grid)
        pixel_reward = pixel_matches * 0.1

        block_2x2_reward = 0
        for y in range(self.GRID_DIM - 1):
            for x in range(self.GRID_DIM - 1):
                if np.array_equal(grid[y:y+2, x:x+2], self.target_grid[y:y+2, x:x+2]):
                    block_2x2_reward += 5
        
        block_3x3_reward = 0
        for y in range(self.GRID_DIM - 2):
            for x in range(self.GRID_DIM - 2):
                if np.array_equal(grid[y:y+3, x:x+3], self.target_grid[y:y+3, x:x+3]):
                    block_3x3_reward += 10
        
        return pixel_reward + block_2x2_reward + block_3x3_reward

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        
        action_taken = movement != 0 or space_held or shift_held

        # 1. Handle selection changes (does not consume a move)
        if space_held:
            self.selected_idx = (self.selected_idx + 1) % (self.GRID_DIM * self.GRID_DIM)
        if shift_held:
            self.selected_idx = (self.selected_idx - 1 + (self.GRID_DIM * self.GRID_DIM)) % (self.GRID_DIM * self.GRID_DIM)

        # 2. Handle movement (consumes a move)
        if movement != 0:
            self.moves_left -= 1
            
            y, x = self.selected_idx // self.GRID_DIM, self.selected_idx % self.GRID_DIM
            ny, nx = y, x

            if movement == 1: ny = (y - 1 + self.GRID_DIM) % self.GRID_DIM # Up
            elif movement == 2: ny = (y + 1) % self.GRID_DIM # Down
            elif movement == 3: nx = (x - 1 + self.GRID_DIM) % self.GRID_DIM # Left
            elif movement == 4: nx = (x + 1) % self.GRID_DIM # Right
            
            # Calculate reward based on change in match score
            score_before = self._calculate_match_reward(self.current_grid)
            
            # Swap pixels
            # # sound: pixel_swap.wav
            self.current_grid[y, x], self.current_grid[ny, nx] = self.current_grid[ny, nx], self.current_grid[y, x]
            
            score_after = self._calculate_match_reward(self.current_grid)
            reward += (score_after - score_before)
            
            # Update selected pixel to the new location
            self.selected_idx = ny * self.GRID_DIM + nx

        # 3. Check for termination conditions
        is_match = np.array_equal(self.current_grid, self.target_grid)
        
        if is_match:
            # # sound: win.wav
            reward += 100
            self.game_over = True
            self.win = True
            terminated = True
            self.level_complexity += 1
        elif self.moves_left <= 0:
            # # sound: lose.wav
            reward -= 50
            self.game_over = True
            self.win = False
            terminated = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            if not terminated: # Only apply penalty if not already won/lost
                reward -= 50
            self.game_over = True
            terminated = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _render_grid(self, grid, top_left, pixel_size, selected_coords=None):
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                color_idx = grid[y, x]
                color = self.PALETTE[color_idx]
                rect = pygame.Rect(
                    top_left[0] + x * pixel_size,
                    top_left[1] + y * pixel_size,
                    pixel_size, pixel_size
                )
                pygame.draw.rect(self.screen, color, rect)

                if selected_coords and (x, y) == selected_coords:
                    # Pulsating glow effect for selection
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
                    alpha = 70 + int(pulse * 80)
                    glow_color = (*self.COLOR_SELECT, alpha)
                    
                    # Use a slightly larger rect for the glow to bleed out
                    glow_rect = rect.inflate(pixel_size * 0.4, pixel_size * 0.4)
                    
                    # Create a temporary surface for alpha blending
                    glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                    pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=int(pixel_size*0.3))
                    self.screen.blit(glow_surf, glow_rect.topleft)

                    # Solid border for clarity
                    pygame.draw.rect(self.screen, self.COLOR_SELECT, rect, 3)

    def _render_game(self):
        # Draw background for main grid
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
            (self.GRID_TOP_LEFT[0], self.GRID_TOP_LEFT[1], self.GRID_AREA_SIZE, self.GRID_AREA_SIZE))
        
        # Draw main grid
        selected_y, selected_x = self.selected_idx // self.GRID_DIM, self.selected_idx % self.GRID_DIM
        self._render_grid(self.current_grid, self.GRID_TOP_LEFT, self.GRID_PIXEL_SIZE, (selected_x, selected_y))

        # Draw target preview
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
            (self.TARGET_TOP_LEFT[0], self.TARGET_TOP_LEFT[1], self.TARGET_AREA_SIZE, self.TARGET_AREA_SIZE))
        self._render_grid(self.target_grid, self.TARGET_TOP_LEFT, self.TARGET_PIXEL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, 
            (self.TARGET_TOP_LEFT[0], self.TARGET_TOP_LEFT[1], self.TARGET_AREA_SIZE, self.TARGET_AREA_SIZE), 2)


    def _render_ui(self):
        # Moves Left
        moves_text = self.FONT_UI.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (20, 20))

        # Score
        score_text = self.FONT_UI.render(f"Score: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 50))
        
        # Level
        level_text = self.FONT_UI.render(f"Level: {self.level_complexity}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (20, 80))

        # Target label
        target_label = self.FONT_SMALL.render("Target", True, self.COLOR_UI_TEXT)
        self.screen.blit(target_label, (self.TARGET_TOP_LEFT[0], self.TARGET_TOP_LEFT[1] + self.TARGET_AREA_SIZE + 5))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg_text = self.FONT_BIG.render("SUCCESS!", True, self.COLOR_SUCCESS)
            else:
                msg_text = self.FONT_BIG.render("OUT OF MOVES", True, self.COLOR_FAIL)
            
            text_rect = msg_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(msg_text, text_rect)

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
            "level_complexity": self.level_complexity,
            "win": self.win,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 800, 600 # Upscale for better viewing
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pixel Shifter")
    
    running = True
    terminated = False
    
    # Map pygame keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement = 0
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                
                if not terminated:
                    # These actions trigger a step
                    if event.key in key_to_action:
                        movement = key_to_action[event.key]
                    if event.key == pygame.K_SPACE:
                        space = 1
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        shift = 1
        
        # Only step if an action was taken or if the game ended
        if movement or space or shift or terminated:
            if not terminated:
                action = np.array([movement, space, shift])
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.scale(surf, (screen_width, screen_height))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(30) # Limit FPS for the player loop

    env.close()