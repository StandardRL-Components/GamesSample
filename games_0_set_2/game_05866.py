import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the block."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Place colored blocks to match adjacent colors. Clear the board before time runs out or it fills up!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Set headless mode for pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        
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
        self.grid_size = 10
        self.max_steps = 1000
        self.fps = 30
        self.initial_time = 30 * self.fps # 30 seconds

        # Visuals
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (60, 70, 80)
        self.COLOR_UI_BG = (40, 45, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        
        self.BLOCK_COLORS = {
            1: (227, 85, 85),   # Red
            2: (85, 194, 227),  # Blue
            3: (85, 227, 135),  # Green
            4: (227, 219, 85),  # Yellow
            5: (188, 85, 227),  # Purple
        }
        self.num_colors = len(self.BLOCK_COLORS)

        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 64)
        
        self.grid_area_height = self.screen_height
        self.cell_size = self.grid_area_height // self.grid_size
        self.grid_width = self.cell_size * self.grid_size
        self.grid_height = self.cell_size * self.grid_size
        self.grid_offset_x = (self.screen_width - self.grid_width) // 2
        self.grid_offset_y = (self.screen_height - self.grid_height) // 2

        # State variables are initialized in reset()
        self.np_random = np.random.default_rng()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.cursor_pos = [0, 0]
        self.timer = 0
        self.next_block_color = 1
        self.prev_space_held = False
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]
        self.timer = self.initial_time
        
        self._generate_new_block()
        
        self.prev_space_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Small penalty for taking a step
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused per brief

        # Update game logic
        self.steps += 1
        self.timer -= 1
        self._update_particles()
        
        # --- Handle player input ---
        # 1. Movement
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.grid_size - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.grid_size - 1, self.cursor_pos[0] + 1)
            
        # 2. Placement (on rising edge of space press)
        place_action = space_held and not self.prev_space_held
        if place_action:
            x, y = self.cursor_pos
            if self.grid[y, x] == 0:
                # Place block
                self.grid[y, x] = self.next_block_color
                
                # Check for matches and clear
                cleared_count, bonus_achieved = self._check_and_clear_matches(x, y)
                
                if cleared_count > 0:
                    reward += cleared_count # +1 per block
                    self.score += cleared_count
                    if bonus_achieved:
                        reward += 5
                        self.score += 5
                
                self._generate_new_block()

        self.prev_space_held = space_held

        # --- Check termination conditions ---
        terminated = False
        truncated = False
        if np.all(self.grid == 0) and self.score > 0: # Win condition
            reward += 100
            self.game_over = True
            terminated = True
            self.win_message = "BOARD CLEARED!"
        elif self.timer <= 0: # Loss: Time up
            reward -= 50
            self.game_over = True
            terminated = True
            self.win_message = "TIME'S UP!"
        elif np.all(self.grid != 0): # Loss: Board full
            reward -= 50
            self.game_over = True
            terminated = True
            self.win_message = "BOARD FULL!"
        
        if self.steps >= self.max_steps: # Truncation
            truncated = True
            self.game_over = True
            self.win_message = "MAX STEPS REACHED"
            
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_new_block(self):
        self.next_block_color = self.np_random.integers(1, self.num_colors + 1)

    def _check_and_clear_matches(self, x, y):
        color_to_match = self.grid[y, x]
        if color_to_match == 0:
            return 0, False

        q = [(x, y)]
        visited = set([(x, y)])
        
        while q:
            cx, cy = q.pop(0)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and (nx, ny) not in visited:
                    if self.grid[ny, nx] == color_to_match:
                        visited.add((nx, ny))
                        q.append((nx, ny))

        if len(visited) > 1:
            for vx, vy in visited:
                self.grid[vy, vx] = 0
                self._spawn_particles(vx, vy, self.BLOCK_COLORS[color_to_match])
            return len(visited), len(visited) > 3
        
        return 0, False
        
    def _spawn_particles(self, grid_x, grid_y, color):
        px = self.grid_offset_x + grid_x * self.cell_size + self.cell_size // 2
        py = self.grid_offset_y + grid_y * self.cell_size + self.cell_size // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(3, 7)
            lifetime = self.np_random.integers(15, 30)
            self.particles.append([ [px, py], vel, size, color, lifetime ])

    def _update_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x
            p[0][1] += p[1][1] # pos.y
            p[4] -= 1 # lifetime
            p[2] *= 0.95 # shrink size
        self.particles = [p for p in self.particles if p[4] > 0 and p[2] > 0.5]
    
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
            "time_left_seconds": self.timer / self.fps,
        }

    def _render_game(self):
        # Draw grid and blocks
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    self.grid_offset_x + x * self.cell_size,
                    self.grid_offset_y + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                
                color_id = self.grid[y, x]
                if color_id > 0:
                    color = self.BLOCK_COLORS[color_id]
                    pygame.gfxdraw.box(self.screen, rect, color)
                    # Inner highlight for 3D effect
                    highlight_color = tuple(min(255, c + 30) for c in color)
                    # FIX: rect.inflate requires two arguments (x, y)
                    pygame.draw.rect(self.screen, highlight_color, rect.inflate(-self.cell_size*0.7, -self.cell_size*0.7), 0, border_radius=int(self.cell_size*0.1))

                pygame.gfxdraw.rectangle(self.screen, rect, self.COLOR_GRID)

        # Draw particles
        for pos, vel, size, color, lifetime in self.particles:
            alpha = int(255 * (lifetime / 30))
            s = int(size)
            if s > 0:
                particle_surf = pygame.Surface((s*2, s*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, color + (alpha,), (s, s), s)
                self.screen.blit(particle_surf, (pos[0] - s, pos[1] - s), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cx * self.cell_size,
            self.grid_offset_y + cy * self.cell_size,
            self.cell_size, self.cell_size
        )
        # Pulsing effect for cursor
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 100 + 155 * pulse
        try:
            pygame.gfxdraw.rectangle(self.screen, cursor_rect, self.COLOR_CURSOR + (int(alpha),))
        except TypeError: # Handle potential alpha issues on some pygame versions
            pygame.gfxdraw.rectangle(self.screen, cursor_rect, self.COLOR_CURSOR)

        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=2)
        
    def _render_ui(self):
        # UI background
        ui_rect = pygame.Rect(0, 0, self.grid_offset_x, self.screen_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)

        # Score display
        score_text = self.font_main.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_rect.centerx - score_text.get_width() // 2, 30))
        self.screen.blit(score_val, (ui_rect.centerx - score_val.get_width() // 2, 55))

        # Timer display
        time_left = max(0, self.timer / self.fps)
        timer_text = self.font_main.render(f"TIME", True, self.COLOR_TEXT)
        timer_val = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (ui_rect.centerx - timer_text.get_width() // 2, 140))
        self.screen.blit(timer_val, (ui_rect.centerx - timer_val.get_width() // 2, 165))

        # Next block display
        next_text = self.font_main.render(f"NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (ui_rect.centerx - next_text.get_width() // 2, 250))
        
        block_rect = pygame.Rect(0, 0, self.cell_size * 1.5, self.cell_size * 1.5)
        block_rect.center = (ui_rect.centerx, 305)
        color = self.BLOCK_COLORS[self.next_block_color]
        pygame.gfxdraw.box(self.screen, block_rect, color)
        highlight_color = tuple(min(255, c + 30) for c in color)
        # FIX: rect.inflate requires two arguments (x, y)
        pygame.draw.rect(self.screen, highlight_color, block_rect.inflate(-block_rect.width*0.7, -block_rect.width*0.7), 0, border_radius=int(block_rect.width*0.1))
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_text, text_rect)

    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # This block will not be run during verification, but is useful for testing
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Test reset
    assert obs.shape == (400, 640, 3)
    assert isinstance(info, dict)
    print("✓ Reset test passed.")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (400, 640, 3)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    print("✓ Step test passed.")

    # Test random agent survival
    done = False
    total_reward = 0
    step_count = 0
    obs, info = env.reset(seed=42)
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated
    
    print(f"Random agent survived for {step_count} steps.")
    
    # Test specific placement and clear
    env.reset(seed=1)
    # Manually set up a clearable scenario
    env.grid = np.zeros((10, 10), dtype=int)
    env.grid[5, 5] = 1 # A red block
    env.next_block_color = 1 # Next block is also red
    env.cursor_pos = [4, 5] # Cursor next to the existing block
    
    # Action: No movement, press space
    action = [0, 1, 0] 
    obs, reward, terminated, truncated, info = env.step(action)
    
    # After this step, the block at (5,5) and the new one at (4,5) should be cleared
    assert env.grid[5, 5] == 0 and env.grid[5, 4] == 0, "Block clearing failed."
    # Reward should be: -0.1 (step) + 2 (cleared blocks) = 1.9
    assert math.isclose(reward, 1.9), f"Reward for clearing 2 blocks is incorrect. Expected 1.9, got {reward}"
    print("✓ Block placement and clearing test passed.")
    
    print("\nAll tests passed!")
    env.close()