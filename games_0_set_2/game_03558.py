import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move the selected pixel. Space/Shift to cycle through pixels."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target image by moving pixels on the grid before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.CELL_SIZE = 40
        self.PIXEL_RADIUS = self.CELL_SIZE // 2 - 4
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X_OFFSET = 40
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        self.UI_X_OFFSET = self.GRID_X_OFFSET + self.GRID_WIDTH + 40
        self.MAX_STEPS = 600

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_SELECT_GLOW = (255, 255, 255)
        self.PIXEL_COLORS = [
            (255, 87, 34),   # Deep Orange
            (255, 193, 7),   # Amber
            (76, 175, 80),   # Green
            (33, 150, 243),  # Blue
            (156, 39, 176),  # Purple
            (233, 30, 99),   # Pink
            (0, 188, 212),   # Cyan
            (139, 195, 74),  # Lime
            (255, 235, 59),  # Yellow
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        self.font_title = pygame.font.Font(None, 24)

        # Game state variables are initialized in reset()
        self.pixels = []
        self.selected_pixel_index = 0
        self.steps = 0
        self.score = 0
        self.correctly_placed_ids = set()
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for submission

    def _generate_puzzle(self):
        """Creates the target and initial shuffled state for the pixels."""
        # A simple smiley face pattern
        target_grid_coords = [
            (2, 2), (3, 2), (6, 2), (7, 2), # Eyes
            (2, 5), (3, 5), (4, 5), (5, 5), (6, 5) # Mouth
        ]

        # Ensure we have enough colors
        num_pixels = len(target_grid_coords)
        assert num_pixels <= len(self.PIXEL_COLORS)

        all_possible_cells = [(c, r) for c in range(self.GRID_COLS) for r in range(self.GRID_ROWS)]
        
        # Ensure the start is not the solved state
        while True:
            start_grid_coords = random.sample(all_possible_cells, k=num_pixels)
            if start_grid_coords != target_grid_coords:
                break

        self.pixels = []
        colors = self.PIXEL_COLORS[:num_pixels]
        for i in range(num_pixels):
            pixel = {
                "id": i,
                "target_pos": target_grid_coords[i],
                "current_pos": start_grid_coords[i],
                "color": colors[i]
            }
            self.pixels.append(pixel)
            
    def _grid_to_screen(self, grid_pos):
        """Converts grid coordinates (col, row) to screen coordinates (x, y)."""
        col, row = grid_pos
        x = self.GRID_X_OFFSET + col * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y_OFFSET + row * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self._generate_puzzle()
        self.steps = 0
        self.score = 0
        self.selected_pixel_index = 0
        self.correctly_placed_ids = set()
        for p in self.pixels:
            if p["current_pos"] == p["target_pos"]:
                self.correctly_placed_ids.add(p["id"])

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # --- 1. Unpack and handle actions ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # Handle selection cycling first
        if space_pressed:
            self.selected_pixel_index = (self.selected_pixel_index + 1) % len(self.pixels)
        elif shift_pressed:
            self.selected_pixel_index = (self.selected_pixel_index - 1 + len(self.pixels)) % len(self.pixels)

        # Handle movement of the selected pixel
        selected_pixel = self.pixels[self.selected_pixel_index]
        old_pos = selected_pixel["current_pos"]
        new_pos = list(old_pos)

        if movement == 1: new_pos[1] -= 1  # Up
        elif movement == 2: new_pos[1] += 1 # Down
        elif movement == 3: new_pos[0] -= 1 # Left
        elif movement == 4: new_pos[0] += 1 # Right
        
        new_pos = tuple(new_pos)

        # --- 2. Validate and update state ---
        is_valid_move = True
        # Check bounds
        if not (0 <= new_pos[0] < self.GRID_COLS and 0 <= new_pos[1] < self.GRID_ROWS):
            is_valid_move = False
        # Check collision
        if any(p["current_pos"] == new_pos for p in self.pixels):
            is_valid_move = False

        if movement != 0 and is_valid_move:
            selected_pixel["current_pos"] = new_pos
            
            # --- 3. Calculate reward ---
            def manhattan_distance(p1, p2):
                return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

            dist_before = manhattan_distance(old_pos, selected_pixel["target_pos"])
            dist_after = manhattan_distance(new_pos, selected_pixel["target_pos"])

            # Continuous reward for getting closer
            reward += 0.1 * (dist_before - dist_after)

            # Event-based reward for correct placement
            if dist_after == 0 and selected_pixel["id"] not in self.correctly_placed_ids:
                reward += 10.0
                self.correctly_placed_ids.add(selected_pixel["id"])
            
            # Update placement status if moved away from target
            if dist_before == 0 and dist_after > 0:
                self.correctly_placed_ids.discard(selected_pixel["id"])
        elif movement != 0 and not is_valid_move:
            pass # No movement, no reward change

        # --- 4. Check for termination ---
        self.steps += 1
        
        is_win = len(self.correctly_placed_ids) == len(self.pixels)
        is_timeout = self.steps >= self.MAX_STEPS

        if is_win:
            reward += 100.0
            terminated = True
            self.score += 100
        elif is_timeout:
            reward -= 50.0
            terminated = True
            self.score -= 50
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))

        # Draw pixels
        for i, p in enumerate(self.pixels):
            screen_pos = self._grid_to_screen(p["current_pos"])
            color = p["color"]
            
            # Draw filled circle for the pixel
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.PIXEL_RADIUS, color)
            pygame.gfxdraw.filled_circle(self.screen, screen_pos[0], screen_pos[1], self.PIXEL_RADIUS, color)
            
            # Draw a subtle highlight if it's in the correct place
            if p["id"] in self.correctly_placed_ids:
                pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], self.PIXEL_RADIUS + 2, (255, 255, 255, 100))

        # Draw selection highlight
        if self.pixels:
            selected_pixel = self.pixels[self.selected_pixel_index]
            screen_pos = self._grid_to_screen(selected_pixel["current_pos"])
            pulse = (math.sin(self.steps * 0.3) + 1) / 2  # 0 to 1
            radius = int(self.PIXEL_RADIUS + 3 + pulse * 2)
            alpha = int(100 + pulse * 155)
            pygame.gfxdraw.aacircle(self.screen, screen_pos[0], screen_pos[1], radius, self.COLOR_SELECT_GLOW + (alpha,))

    def _render_ui(self):
        # --- Score Display ---
        score_text = self.font_large.render(f"{int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(midtop=(self.UI_X_OFFSET + 80, self.GRID_Y_OFFSET))
        self.screen.blit(score_text, score_rect)
        score_label = self.font_title.render("SCORE", True, self.COLOR_GRID)
        score_label_rect = score_label.get_rect(midtop=(score_rect.centerx, score_rect.bottom))
        self.screen.blit(score_label, score_label_rect)
        
        # --- Steps/Timer Display ---
        steps_text = self.font_large.render(f"{self.MAX_STEPS - self.steps}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(midtop=(self.UI_X_OFFSET + 80, self.GRID_Y_OFFSET + 80))
        self.screen.blit(steps_text, steps_rect)
        steps_label = self.font_title.render("TIME LEFT", True, self.COLOR_GRID)
        steps_label_rect = steps_label.get_rect(midtop=(steps_rect.centerx, steps_rect.bottom))
        self.screen.blit(steps_label, steps_label_rect)

        # --- Target Preview ---
        preview_label = self.font_title.render("TARGET", True, self.COLOR_GRID)
        self.screen.blit(preview_label, (self.UI_X_OFFSET, self.GRID_Y_OFFSET + 160))
        
        preview_cell_size = 12
        preview_x_start = self.UI_X_OFFSET
        preview_y_start = self.GRID_Y_OFFSET + 185
        
        # Draw preview grid
        for r in range(self.GRID_ROWS + 1):
            y = preview_y_start + r * preview_cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (preview_x_start, y), (preview_x_start + self.GRID_COLS * preview_cell_size, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = preview_x_start + c * preview_cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, preview_y_start), (x, preview_y_start + self.GRID_ROWS * preview_cell_size), 1)
            
        # Draw target pixels in preview
        for p in self.pixels:
            col, row = p["target_pos"]
            color = p["color"]
            rect = pygame.Rect(
                preview_x_start + col * preview_cell_size + 1,
                preview_y_start + row * preview_cell_size + 1,
                preview_cell_size - 1,
                preview_cell_size - 1
            )
            pygame.draw.rect(self.screen, color, rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "pixels_correct": len(self.correctly_placed_ids),
            "pixels_total": len(self.pixels)
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pixel Puzzle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.game_description)
    print(env.user_guide)

    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        # A simple debounce mechanism for human play
        if 'last_action_time' not in locals():
            last_action_time = 0
        
        key_pressed_this_frame = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                key_pressed_this_frame = True
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("--- Game Reset ---")
                
                # Handle single-press actions
                if event.key == pygame.K_SPACE: action[1] = 1
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: action[2] = 1

        # Get key states for continuous actions (movement)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: 
            action[0] = 1
            key_pressed_this_frame = True
        elif keys[pygame.K_DOWN]: 
            action[0] = 2
            key_pressed_this_frame = True
        elif keys[pygame.K_LEFT]: 
            action[0] = 3
            key_pressed_this_frame = True
        elif keys[pygame.K_RIGHT]: 
            action[0] = 4
            key_pressed_this_frame = True
        
        current_time = pygame.time.get_ticks()
        
        if key_pressed_this_frame and current_time - last_action_time > 150: # 150ms delay
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            last_action_time = current_time

            if terminated:
                print("--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Final Steps: {info['steps']}")
                # Optional: auto-reset after a delay
                pygame.time.wait(2000)
                obs, info = env.reset()
                total_reward = 0

        # Render the observation to the display window
        # The observation is (H, W, C), but pygame surface wants (W, H)
        # So we need to transpose it back
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS for human play

    env.close()