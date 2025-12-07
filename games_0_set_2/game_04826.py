
# Generated: 2025-08-28T03:09:01.453473
# Source Brief: brief_04826.md
# Brief Index: 4826

        
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
        "Controls: Use arrow keys to move the cursor. Hold SHIFT and use LEFT/RIGHT to change color. Press SPACE to paint."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Recreate the target pixel art before you run out of time or paint. Place colors strategically to match the reference image."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 22)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINES = (50, 60, 80)
        self.COLOR_EMPTY = (35, 40, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 230)
        self.PAINT_COLORS = [
            (239, 71, 111),  # Red
            (255, 209, 102), # Yellow
            (6, 214, 160),   # Green
            (17, 138, 178),  # Blue
            (7, 59, 76),     # Dark Blue
            (255, 255, 255)  # White
        ]
        self.TARGET_COLORS = [pygame.Color(c).lerp(self.COLOR_BG, 0.4) for c in self.PAINT_COLORS]

        # --- Game Constants ---
        self.GRID_DIM = 10
        self.MAX_STEPS = 600
        self.INITIAL_PAINT = 15
        
        # --- Game State (initialized in reset) ---
        self.grid = None
        self.target_grid = None
        self.paint_counts = None
        self.cursor_pos = None
        self.selected_color_idx = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_over_message = ""
        self.last_space_held = None
        self.rewarded_rows = None
        self.rewarded_cols = None
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._generate_target()
        self.grid = np.full((self.GRID_DIM, self.GRID_DIM), -1, dtype=int)
        self.paint_counts = np.full(len(self.PAINT_COLORS), self.INITIAL_PAINT, dtype=int)
        
        self.cursor_pos = [0, 0]
        self.selected_color_idx = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.last_space_held = False
        self.rewarded_rows = set()
        self.rewarded_cols = set()
        self.particles = []

        return self._get_observation(), self._get_info()

    def _generate_target(self):
        while True:
            # Generate a random grid, biased towards being empty (-1)
            target = self.np_random.integers(-3, len(self.PAINT_COLORS), size=(self.GRID_DIM, self.GRID_DIM))
            target[target < 0] = -1

            # Count paint requirements
            colors, counts = np.unique(target[target != -1], return_counts=True)
            paint_req = np.zeros(len(self.PAINT_COLORS), dtype=int)
            if len(colors) > 0:
                paint_req[colors] = counts

            # Ensure it's solvable and not trivial
            if np.all(paint_req <= self.INITIAL_PAINT) and np.sum(paint_req) > 10:
                self.target_grid = target
                return

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.steps += 1

        # --- Handle Input ---
        self._handle_input(movement, space_held, shift_held)
        
        # --- Place Pixel Action ---
        if space_held and not self.last_space_held:
            reward += self._place_pixel()

        self.last_space_held = space_held

        # --- Check Termination Conditions ---
        term_reward, terminated = self._check_termination()
        reward += term_reward
        if terminated:
            self.game_over = True

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Color selection (Shift + Left/Right)
        if shift_held:
            if movement == 3: # Left
                self.selected_color_idx = (self.selected_color_idx - 1) % len(self.PAINT_COLORS)
            elif movement == 4: # Right
                self.selected_color_idx = (self.selected_color_idx + 1) % len(self.PAINT_COLORS)
        # Cursor movement
        else:
            if movement == 1: # Up
                self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: # Down
                self.cursor_pos[1] = min(self.GRID_DIM - 1, self.cursor_pos[1] + 1)
            elif movement == 3: # Left
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: # Right
                self.cursor_pos[0] = min(self.GRID_DIM - 1, self.cursor_pos[0] + 1)

    def _place_pixel(self):
        x, y = self.cursor_pos
        reward = 0

        # Can't paint if out of the selected color
        if self.paint_counts[self.selected_color_idx] <= 0:
            # sfx: empty_click
            return 0
        
        # Don't penalize for repainting an already correct square
        if self.grid[y, x] == self.target_grid[y, x]:
            return 0

        # Paint the square
        # sfx: paint_splat
        self.grid[y, x] = self.selected_color_idx
        self.paint_counts[self.selected_color_idx] -= 1
        
        # Add particle effect
        px, py = 185 + x * 30, 100 + y * 30
        self._create_particle(px, py, self.PAINT_COLORS[self.selected_color_idx])

        # Calculate placement reward
        if self.grid[y, x] == self.target_grid[y, x]:
            reward += 0.1
        else:
            reward -= 0.01
        
        # Check for row/column completion rewards
        reward += self._check_line_completion(x, y)
        
        return reward

    def _check_line_completion(self, x, y):
        reward = 0
        # Check row
        if y not in self.rewarded_rows:
            if np.array_equal(self.grid[y, :], self.target_grid[y, :]):
                reward += 5
                self.rewarded_rows.add(y)
                # sfx: line_complete
        # Check column
        if x not in self.rewarded_cols:
            if np.array_equal(self.grid[:, x], self.target_grid[:, x]):
                reward += 5
                self.rewarded_cols.add(x)
                # sfx: line_complete
        return reward

    def _check_termination(self):
        # 1. Win condition
        if np.array_equal(self.grid, self.target_grid):
            self.game_over_message = "PERFECT!"
            # sfx: win_fanfare
            return 100, True

        # 2. Time limit
        if self.steps >= self.MAX_STEPS:
            self.game_over_message = "TIME'S UP!"
            # sfx: lose_buzzer
            return -25, True

        # 3. Unwinnable state (out of required paint)
        is_unwinnable = False
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                target_color = self.target_grid[r, c]
                if self.grid[r, c] != target_color and target_color != -1:
                    if self.paint_counts[target_color] <= 0:
                        is_unwinnable = True
                        break
            if is_unwinnable:
                break
        
        if is_unwinnable:
            self.game_over_message = "OUT OF PAINT!"
            # sfx: lose_sad
            return -50, True

        return 0, False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # --- Grid & Cells ---
        grid_rect = pygame.Rect(170, 85, 300, 300)
        cell_size = 30
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                color_idx = self.grid[r, c]
                color = self.PAINT_COLORS[color_idx] if color_idx != -1 else self.COLOR_EMPTY
                cell_rect = pygame.Rect(grid_rect.left + c * cell_size, grid_rect.top + r * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, color, cell_rect)
        
        # --- Grid Lines ---
        for i in range(self.GRID_DIM + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (grid_rect.left, grid_rect.top + i * cell_size), (grid_rect.right, grid_rect.top + i * cell_size))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (grid_rect.left + i * cell_size, grid_rect.top), (grid_rect.left + i * cell_size, grid_rect.bottom))

        # --- Particles ---
        self._update_and_draw_particles()

        # --- Cursor ---
        if not self.game_over:
            cursor_x = grid_rect.left + self.cursor_pos[0] * cell_size
            cursor_y = grid_rect.top + self.cursor_pos[1] * cell_size
            pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
            thickness = int(1 + pulse * 2)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, cell_size, cell_size), thickness)

    def _render_ui(self):
        # --- Target Image ---
        target_size = 12
        target_rect = pygame.Rect((self.screen_width - self.GRID_DIM * target_size) / 2, 15, self.GRID_DIM * target_size, self.GRID_DIM * target_size)
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                color_idx = self.target_grid[r, c]
                color = self.TARGET_COLORS[color_idx] if color_idx != -1 else self.COLOR_EMPTY
                cell_rect = pygame.Rect(target_rect.left + c * target_size, target_rect.top + r * target_size, target_size, target_size)
                pygame.draw.rect(self.screen, color, cell_rect)
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, target_rect, 1)
        
        # --- Palette & Paint Counts ---
        swatch_size = 25
        swatch_spacing = 8
        palette_start_x = 20
        for i, color in enumerate(self.PAINT_COLORS):
            x = palette_start_x + i * (swatch_size + swatch_spacing)
            y = self.screen_height - 45
            
            # Draw swatch
            swatch_rect = pygame.Rect(x, y, swatch_size, swatch_size)
            pygame.draw.rect(self.screen, color, swatch_rect)
            
            # Highlight selected
            if i == self.selected_color_idx and not self.game_over:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, swatch_rect, 2)
            
            # Draw count
            count_text = self.font_small.render(str(self.paint_counts[i]), True, self.COLOR_TEXT)
            self.screen.blit(count_text, (x + swatch_size/2 - count_text.get_width()/2, y + swatch_size + 3))

        # --- Score & Time ---
        score_text = self.font_medium.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 20, self.screen_height - 65))
        
        time_left = self.MAX_STEPS - self.steps
        time_color = (230, 80, 80) if time_left < 100 else self.COLOR_TEXT
        time_text = self.font_medium.render(f"STEPS: {time_left}", True, time_color)
        self.screen.blit(time_text, (self.screen_width - time_text.get_width() - 20, self.screen_height - 35))

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 200))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_over_message, True, self.COLOR_CURSOR)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _create_particle(self, x, y, color):
        for _ in range(10):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            particle = {
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'life': random.uniform(10, 20),
                'color': color
            }
            self.particles.append(particle)

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, color)

    def validate_implementation(self):
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Pixel Painter")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    running = True
    
    # Track pressed keys for single action per press
    last_action = [0, 0, 0]

    while running:
        action_taken_this_frame = False
        
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action_taken_this_frame = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
            if event.type == pygame.KEYUP:
                action_taken_this_frame = True
        
        # --- Step Environment ---
        # Since auto_advance is False, we only step when an action occurs.
        if not terminated and action_taken_this_frame:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        # We always get the latest observation, even if we didn't step
        latest_obs = env._get_observation()
        surf = pygame.surfarray.make_surface(np.transpose(latest_obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate

    pygame.quit()