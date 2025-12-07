
# Generated: 2025-08-27T20:28:00.195829
# Source Brief: brief_02468.md
# Brief Index: 2468

        
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
        "Controls: ↑↓←→ to move cursor. Tap Shift to cycle colors. Tap Space to place a color."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image on your canvas. You have a limited number of placements to reach 95% accuracy."
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
        self.GRID_SIZE = (10, 10)
        self.MAX_PLACEMENTS = 300
        self.WIN_THRESHOLD = 0.95
        self.PIXEL_SIZE = 20
        
        self._define_colors_and_fonts()
        self._define_patterns()
        
        # State variables are initialized in reset()
        self.target_image = None
        self.player_grid = None
        self.palette = None
        self.blank_color_idx = 0
        self.cursor_pos = [0, 0]
        self.selected_color_idx = 0
        self.placements_left = 0
        self.accuracy = 0.0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.prev_shift_held = False
        self.prev_space_held = False
        self.placement_flash = 0  # For visual feedback
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def _define_colors_and_fonts(self):
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_BG = (43, 48, 61)
        self.COLOR_GRID_LINE = (62, 68, 81)
        self.COLOR_TEXT = (222, 222, 222)
        self.COLOR_TEXT_ACCENT = (138, 226, 153)
        self.COLOR_CURSOR = (252, 233, 79)
        self.COLOR_WIN = (114, 235, 121)
        self.COLOR_LOSE = (239, 65, 65)
        self.font_s = pygame.font.Font(None, 20)
        self.font_m = pygame.font.Font(None, 28)
        self.font_l = pygame.font.Font(None, 48)

    def _define_patterns(self):
        # Patterns are defined by a grid of indices and a corresponding palette.
        # Index 0 in the palette is always the "blank" color.
        self.PATTERNS = {
            "smiley": {
                "palette": [(50, 50, 60), (255, 230, 90), (30, 30, 30)],
                "grid": np.array([
                    [0,0,1,1,1,1,1,1,0,0],
                    [0,1,1,1,1,1,1,1,1,0],
                    [1,1,2,1,1,1,1,2,1,1],
                    [1,1,1,1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1],
                    [1,1,2,1,1,1,1,2,1,1],
                    [1,1,1,2,2,2,2,1,1,1],
                    [1,1,1,1,1,1,1,1,1,1],
                    [0,1,1,1,1,1,1,1,1,0],
                    [0,0,1,1,1,1,1,1,0,0],
                ])
            },
            "heart": {
                "palette": [(50, 50, 60), (220, 50, 50), (255, 150, 150)],
                "grid": np.array([
                    [0,0,1,1,0,0,1,1,0,0],
                    [0,1,2,2,1,1,2,2,1,0],
                    [1,2,2,2,2,2,2,2,2,1],
                    [1,2,2,2,2,2,2,2,2,1],
                    [1,2,2,2,2,2,2,2,2,1],
                    [0,1,2,2,2,2,2,2,1,0],
                    [0,0,1,2,2,2,2,1,0,0],
                    [0,0,0,1,2,2,1,0,0,0],
                    [0,0,0,0,1,1,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                ])
            },
            "sword": {
                "palette": [(50, 50, 60), (180, 180, 190), (100, 100, 110), (150, 120, 80)],
                "grid": np.array([
                    [0,0,0,0,0,1,0,0,0,0],
                    [0,0,0,0,1,2,1,0,0,0],
                    [0,0,0,0,1,2,1,0,0,0],
                    [0,0,0,0,1,2,1,0,0,0],
                    [0,0,0,0,1,2,1,0,0,0],
                    [0,0,0,3,2,2,2,3,0,0],
                    [0,0,3,3,2,2,2,3,3,0],
                    [0,0,0,3,3,2,3,3,0,0],
                    [0,0,0,0,3,3,3,0,0,0],
                    [0,0,0,0,0,3,0,0,0,0],
                ])
            }
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_target()
        
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.selected_color_idx = 1 # Start with a non-blank color
        self.placements_left = self.MAX_PLACEMENTS
        self.steps = 0
        self.game_over = False
        self.win = False
        self.prev_shift_held = True # Prevent action on first frame
        self.prev_space_held = True # Prevent action on first frame
        self.placement_flash = 0
        
        self._calculate_accuracy()

        return self._get_observation(), self._get_info()
    
    def _generate_target(self):
        pattern_name = self.np_random.choice(list(self.PATTERNS.keys()))
        pattern = self.PATTERNS[pattern_name]
        self.palette = pattern["palette"]
        self.target_image = pattern["grid"]
        self.blank_color_idx = 0
        self.player_grid = np.full(self.GRID_SIZE, self.blank_color_idx, dtype=np.int8)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Decay visual effects
        if self.placement_flash > 0:
            self.placement_flash -= 1

        reward = 0
        
        # Handle single-press actions (debouncing)
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held

        # Update game logic
        self._handle_movement(movement)
        
        if shift_press:
            self._handle_color_cycle()
            # sfx: color_cycle.wav
        
        if space_press:
            reward = self._handle_placement()
            
        self.steps += 1
        terminated = self._check_termination()

        # Add terminal rewards
        if terminated:
            if self.win:
                reward += 100
                # sfx: win_fanfare.wav
            else:
                reward -= 100
                # sfx: lose_buzzer.wav

        # Update previous action states for debouncing
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE[1] - 1)

    def _handle_color_cycle(self):
        self.selected_color_idx = (self.selected_color_idx + 1) % len(self.palette)
        if self.selected_color_idx == self.blank_color_idx: # Skip blank color
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.palette)

    def _handle_placement(self):
        if self.placements_left <= 0:
            return 0 # No reward if no placements left

        x, y = self.cursor_pos
        
        # Don't waste a placement if the color is already correct
        if self.player_grid[y, x] == self.selected_color_idx:
            return 0

        self.placements_left -= 1
        
        # Check if placement was correct before updating the grid
        is_correct = self.selected_color_idx == self.target_image[y, x]
        
        self.player_grid[y, x] = self.selected_color_idx
        self.placement_flash = 5 # Set flash duration
        
        self._calculate_accuracy()

        if is_correct:
            # sfx: place_correct.wav
            return 1
        else:
            # sfx: place_incorrect.wav
            return -1

    def _calculate_accuracy(self):
        correct_pixels = np.sum(self.player_grid == self.target_image)
        total_pixels = self.GRID_SIZE[0] * self.GRID_SIZE[1]
        self.accuracy = correct_pixels / total_pixels

    def _check_termination(self):
        if self.accuracy >= self.WIN_THRESHOLD:
            self.game_over = True
            self.win = True
            return True
        if self.placements_left <= 0:
            self.game_over = True
            self.win = False
            return True
        return False
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        grid_w = self.GRID_SIZE[0] * self.PIXEL_SIZE
        
        # Draw Target Grid
        target_x = (self.screen.get_width() // 2) - grid_w - 20
        target_y = 80
        self._draw_grid(self.target_image, (target_x, target_y), "TARGET")

        # Draw Player Grid
        player_x = (self.screen.get_width() // 2) + 20
        player_y = 80
        self._draw_grid(self.player_grid, (player_x, player_y), "YOUR CANVAS")
        
        # Draw Cursor
        cursor_x = player_x + self.cursor_pos[0] * self.PIXEL_SIZE
        cursor_y = player_y + self.cursor_pos[1] * self.PIXEL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.PIXEL_SIZE, self.PIXEL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        
        # Draw placement flash
        if self.placement_flash > 0:
            flash_surface = pygame.Surface((self.PIXEL_SIZE, self.PIXEL_SIZE), pygame.SRCALPHA)
            alpha = int(255 * (self.placement_flash / 5))
            flash_surface.fill((255, 255, 255, alpha))
            self.screen.blit(flash_surface, (cursor_x, cursor_y))

    def _draw_grid(self, grid_data, top_left, title):
        x, y = top_left
        grid_w = self.GRID_SIZE[0] * self.PIXEL_SIZE
        grid_h = self.GRID_SIZE[1] * self.PIXEL_SIZE
        
        # Draw title
        title_surf = self.font_m.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (x + grid_w // 2 - title_surf.get_width() // 2, y - 30))

        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x, y, grid_w, grid_h))

        # Draw pixels
        for row in range(self.GRID_SIZE[1]):
            for col in range(self.GRID_SIZE[0]):
                color = self.palette[grid_data[row, col]]
                pygame.draw.rect(self.screen, color, 
                                 (x + col * self.PIXEL_SIZE, y + row * self.PIXEL_SIZE, 
                                  self.PIXEL_SIZE, self.PIXEL_SIZE))
        
        # Draw grid lines
        for i in range(self.GRID_SIZE[0] + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                             (x + i * self.PIXEL_SIZE, y), 
                             (x + i * self.PIXEL_SIZE, y + grid_h))
        for i in range(self.GRID_SIZE[1] + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, 
                             (x, y + i * self.PIXEL_SIZE), 
                             (x + grid_w, y + i * self.PIXEL_SIZE))

    def _render_ui(self):
        # Accuracy display
        acc_text = f"ACCURACY: {self.accuracy:.0%}"
        acc_surf = self.font_m.render(acc_text, True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(acc_surf, (20, self.screen.get_height() - 40))

        # Placements display
        pl_text = f"PLACEMENTS LEFT: {self.placements_left}"
        pl_surf = self.font_m.render(pl_text, True, self.COLOR_TEXT)
        self.screen.blit(pl_surf, (self.screen.get_width() - pl_surf.get_width() - 20, self.screen.get_height() - 40))

        # Selected color display
        swatch_x = self.screen.get_width() // 2 - 50
        swatch_y = self.screen.get_height() - 55
        
        sel_text_surf = self.font_s.render("SELECTED", True, self.COLOR_TEXT)
        self.screen.blit(sel_text_surf, (swatch_x + 35, swatch_y))
        
        pygame.draw.rect(self.screen, self.palette[self.selected_color_idx], (swatch_x, swatch_y + 2, 30, 30))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (swatch_x, swatch_y + 2, 30, 30), 1)

    def _render_game_over(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        text = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_WIN if self.win else self.COLOR_LOSE
        
        text_surf = self.font_l.render(text, True, color)
        text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.accuracy * 100, # Use accuracy percentage as score
            "steps": self.steps,
            "placements_left": self.placements_left,
            "win": self.win
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually, you would need a different setup that maps keyboard events.
    # This example just runs a random agent.
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # For visualization with Pygame
    pygame.display.set_caption("Pixel Art Puzzle")
    screen = pygame.display.set_mode((640, 400))
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Random agent logic
        if not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                print(f"Episode finished. Total reward: {total_reward}, Info: {info}")
        
        # Render the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # If done, wait a bit then reset
        if done:
            pygame.time.wait(2000)
            obs, info = env.reset()
            done = False
            total_reward = 0

        # Since auto_advance is False, we need to step to see changes
        # For a smooth random agent view, we can add a small delay
        pygame.time.wait(50)

    env.close()