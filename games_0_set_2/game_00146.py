
# Generated: 2025-08-27T12:44:41.042622
# Source Brief: brief_00146.md
# Brief Index: 146

        
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
        "Controls: Arrow keys to push pixels. Space to cycle selected color. Recreate the target patterns before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced pixel puzzle game. Select a color and push all pixels of that color across the grid to match the target patterns on the right."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 15
        self.CELL_SIZE = 20
        self.T_GRID_SIZE = 5
        self.T_CELL_SIZE = 8
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.FPS * self.TIME_LIMIT_SECONDS

        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID_LINES = (50, 55, 65)
        self.COLOR_UI_TEXT = (220, 220, 230)
        self.COLOR_UI_ACCENT = (100, 200, 255)
        self.COLOR_MATCH_GLOW = (180, 255, 180)
        self.PIXEL_COLORS = [
            (0, 0, 0),  # 0: Empty
            (255, 80, 80),  # 1: Red
            (80, 255, 80),  # 2: Green
            (80, 150, 255), # 3: Blue
            (255, 255, 80), # 4: Yellow
            (200, 80, 255), # 5: Purple
        ]
        self.NUM_COLORS = len(self.PIXEL_COLORS) - 1

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
            self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 14)
        except IOError:
            self.font_large = pygame.font.SysFont("sans", 24)
            self.font_medium = pygame.font.SysFont("sans", 18)
            self.font_small = pygame.font.SysFont("sans", 14)


        # --- Game State ---
        self.grid = None
        self.target_patterns = None
        self.patterns_matched = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_color_idx = 1
        self.prev_space_held = False
        self.match_effects = []

        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_color_idx = 1
        self.prev_space_held = False
        self.match_effects = []
        
        self._generate_patterns()
        self._generate_grid()
        self.patterns_matched = [False] * 5
        self.misplaced_pixels_cache = self._calculate_total_misplaced_pixels()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Handle Actions ---
        # Cycle selected color on space press (rising edge)
        if space_held and not self.prev_space_held:
            # sfx: color_select.wav
            self.selected_color_idx = (self.selected_color_idx % self.NUM_COLORS) + 1
        self.prev_space_held = space_held

        # Perform push action
        if movement != 0:
            # sfx: push.wav
            self._perform_push(movement)

        # --- Update Game Logic ---
        self.steps += 1
        
        # Check for new pattern matches
        newly_matched_reward = self._check_matches()
        reward += newly_matched_reward
        if newly_matched_reward > 0:
            # sfx: match_success.wav
            self.score += newly_matched_reward

        # Calculate continuous reward based on improvement
        current_misplaced = self._calculate_total_misplaced_pixels()
        improvement = self.misplaced_pixels_cache - current_misplaced
        reward += improvement * 0.1
        self.misplaced_pixels_cache = current_misplaced

        # --- Check Termination ---
        time_is_up = self.steps >= self.MAX_STEPS
        all_patterns_matched = all(self.patterns_matched)
        terminated = time_is_up or all_patterns_matched

        if terminated:
            self.game_over = True
            if all_patterns_matched:
                # sfx: victory.wav
                reward += 50  # Victory bonus
                self.score += 50
            else: # Time ran out
                # sfx: failure.wav
                reward -= 50  # Timeout penalty
                self.score -= 50

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info()
        )

    def _generate_patterns(self):
        self.target_patterns = []
        for _ in range(5):
            pattern = self.np_random.integers(0, self.NUM_COLORS + 1, size=(self.T_GRID_SIZE, self.T_GRID_SIZE))
            # Ensure pattern is not empty
            while np.sum(pattern) == 0:
                 pattern = self.np_random.integers(0, self.NUM_COLORS + 1, size=(self.T_GRID_SIZE, self.T_GRID_SIZE))
            self.target_patterns.append(pattern)

    def _generate_grid(self):
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        pixel_counts = {i: 0 for i in range(1, self.NUM_COLORS + 1)}
        for pattern in self.target_patterns:
            for color_idx in range(1, self.NUM_COLORS + 1):
                pixel_counts[color_idx] += np.sum(pattern == color_idx)

        pixels_to_place = []
        for color_idx, count in pixel_counts.items():
            pixels_to_place.extend([color_idx] * count)
        
        # Fill some portion of the grid to ensure solvability
        if len(pixels_to_place) > self.GRID_WIDTH * self.GRID_HEIGHT:
            pixels_to_place = pixels_to_place[:self.GRID_WIDTH * self.GRID_HEIGHT]

        self.np_random.shuffle(pixels_to_place)
        
        empty_cells = self.GRID_WIDTH * self.GRID_HEIGHT - len(pixels_to_place)
        flat_grid = pixels_to_place + [0] * empty_cells
        self.np_random.shuffle(flat_grid)
        
        self.grid = np.array(flat_grid).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))


    def _perform_push(self, direction):
        # 1=up, 2=down, 3=left, 4=right
        if direction == 1: # Up
            for y in range(self.GRID_HEIGHT):
                for x in range(self.GRID_WIDTH):
                    if self.grid[y, x] == self.selected_color_idx:
                        ny = (y - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
                        if self.grid[ny, x] == 0:
                            self.grid[ny, x] = self.selected_color_idx
                            self.grid[y, x] = 0
        elif direction == 2: # Down
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                for x in range(self.GRID_WIDTH):
                    if self.grid[y, x] == self.selected_color_idx:
                        ny = (y + 1) % self.GRID_HEIGHT
                        if self.grid[ny, x] == 0:
                            self.grid[ny, x] = self.selected_color_idx
                            self.grid[y, x] = 0
        elif direction == 3: # Left
            for x in range(self.GRID_WIDTH):
                for y in range(self.GRID_HEIGHT):
                    if self.grid[y, x] == self.selected_color_idx:
                        nx = (x - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
                        if self.grid[y, nx] == 0:
                            self.grid[y, nx] = self.selected_color_idx
                            self.grid[y, x] = 0
        elif direction == 4: # Right
            for x in range(self.GRID_WIDTH - 1, -1, -1):
                for y in range(self.GRID_HEIGHT):
                    if self.grid[y, x] == self.selected_color_idx:
                        nx = (x + 1) % self.GRID_WIDTH
                        if self.grid[y, nx] == 0:
                            self.grid[y, nx] = self.selected_color_idx
                            self.grid[y, x] = 0

    def _check_matches(self):
        reward = 0
        for i, pattern in enumerate(self.target_patterns):
            if not self.patterns_matched[i]:
                p_h, p_w = pattern.shape
                for y in range(self.GRID_HEIGHT - p_h + 1):
                    for x in range(self.GRID_WIDTH - p_w + 1):
                        sub_grid = self.grid[y:y+p_h, x:x+p_w]
                        if np.array_equal(sub_grid, pattern):
                            self.patterns_matched[i] = True
                            reward += 10
                            self.match_effects.append({
                                'grid_pos': (x, y),
                                'target_idx': i,
                                'timer': self.FPS // 2 # 0.5 second glow
                            })
                            break
                    if self.patterns_matched[i]:
                        break
        return reward
    
    def _calculate_total_misplaced_pixels(self):
        total_misplaced = 0
        for i, pattern in enumerate(self.target_patterns):
            if self.patterns_matched[i]:
                continue
            
            p_h, p_w = pattern.shape
            min_misplaced_for_pattern = p_h * p_w

            for y in range(self.GRID_HEIGHT - p_h + 1):
                for x in range(self.GRID_WIDTH - p_w + 1):
                    sub_grid = self.grid[y:y+p_h, x:x+p_w]
                    # Count non-matching pixels where target is not empty
                    misplaced = np.sum((sub_grid != pattern) & (pattern != 0))
                    min_misplaced_for_pattern = min(min_misplaced_for_pattern, misplaced)
            total_misplaced += min_misplaced_for_pattern
        return total_misplaced

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Main Grid ---
        grid_area_x, grid_area_y = 40, 60
        grid_pixel_w = self.GRID_WIDTH * self.CELL_SIZE
        grid_pixel_h = self.GRID_HEIGHT * self.CELL_SIZE

        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = grid_area_x + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, grid_area_y), (x, grid_area_y + grid_pixel_h))
        for i in range(self.GRID_HEIGHT + 1):
            y = grid_area_y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (grid_area_x, y), (grid_area_x + grid_pixel_w, y))

        # Draw pixels
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx > 0:
                    color = self.PIXEL_COLORS[color_idx]
                    rect = pygame.Rect(
                        grid_area_x + x * self.CELL_SIZE + 1,
                        grid_area_y + y * self.CELL_SIZE + 1,
                        self.CELL_SIZE - 1,
                        self.CELL_SIZE - 1
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=3)

        # --- Draw Target Patterns ---
        target_area_x = grid_area_x + grid_pixel_w + 40
        for i, pattern in enumerate(self.target_patterns):
            p_h, p_w = pattern.shape
            offset_y = i * (p_h * self.T_CELL_SIZE + 15)
            base_y = grid_area_y + 10 + offset_y

            # Draw border
            border_rect = pygame.Rect(target_area_x - 5, base_y - 5, p_w * self.T_CELL_SIZE + 10, p_h * self.T_CELL_SIZE + 10)
            border_color = self.COLOR_MATCH_GLOW if self.patterns_matched[i] else self.COLOR_GRID_LINES
            pygame.draw.rect(self.screen, border_color, border_rect, 2, border_radius=5)

            # Draw pattern pixels
            for y in range(p_h):
                for x in range(p_w):
                    color_idx = pattern[y, x]
                    if color_idx > 0:
                        color = self.PIXEL_COLORS[color_idx]
                        rect = pygame.Rect(
                            target_area_x + x * self.T_CELL_SIZE,
                            base_y + y * self.T_CELL_SIZE,
                            self.T_CELL_SIZE,
                            self.T_CELL_SIZE
                        )
                        pygame.draw.rect(self.screen, color, rect, border_radius=1)

        # --- Draw Match Effects ---
        new_effects = []
        for effect in self.match_effects:
            if effect['timer'] > 0:
                # Glow on main grid
                gx, gy = effect['grid_pos']
                p_h, p_w = self.target_patterns[effect['target_idx']].shape
                glow_rect = pygame.Rect(
                    grid_area_x + gx * self.CELL_SIZE,
                    grid_area_y + gy * self.CELL_SIZE,
                    p_w * self.CELL_SIZE,
                    p_h * self.CELL_SIZE
                )
                alpha = int(150 * (effect['timer'] / (self.FPS // 2)))
                s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(s, (*self.COLOR_MATCH_GLOW, alpha), s.get_rect(), border_radius=8)
                self.screen.blit(s, glow_rect.topleft)

                effect['timer'] -= 1
                new_effects.append(effect)
        self.match_effects = new_effects


    def _render_ui(self):
        # --- Top UI Bar ---
        pygame.draw.rect(self.screen, (35, 40, 50), (0, 0, self.WIDTH, 45))
        pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (0, 45), (self.WIDTH, 45))

        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = f"Time: {time_left:.1f}s"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (20, 12))

        # Matches
        matches_count = sum(self.patterns_matched)
        matches_text = f"Matches: {matches_count} / 5"
        matches_surf = self.font_medium.render(matches_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(matches_surf, (180, 12))

        # Selected Color
        color_text = self.font_medium.render("Selected:", True, self.COLOR_UI_TEXT)
        self.screen.blit(color_text, (340, 12))
        color_box_rect = pygame.Rect(440, 10, 25, 25)
        pygame.draw.rect(self.screen, self.PIXEL_COLORS[self.selected_color_idx], color_box_rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, color_box_rect, 2, border_radius=4)

        # Game Over Text
        if self.game_over:
            outcome_text = "ALL PATTERNS MATCHED!" if all(self.patterns_matched) else "TIME'S UP!"
            color = self.COLOR_MATCH_GLOW if all(self.patterns_matched) else (255, 100, 100)
            
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0,0))

            text_surf = self.font_large.render(outcome_text, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS),
            "patterns_matched": sum(self.patterns_matched),
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


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Pusher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        movement = 0 # no-op
        space_held = False
        shift_held = False # Unused in this game

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)

    env.close()