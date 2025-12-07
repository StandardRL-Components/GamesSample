
# Generated: 2025-08-27T15:45:18.847104
# Source Brief: brief_01068.md
# Brief Index: 1068

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a gem group. Match 3 or more to collect."
    )

    game_description = (
        "Race against time to collect matching sets of gems. Select groups of 3 or more to score points and fill your collection goal. Plan your moves to create combos and win before the timer runs out."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = (10, 8)  # Columns, Rows
        self.NUM_GEM_TYPES = 6
        self.MATCH_MIN = 3
        self.WIN_GEMS_TARGET = 20
        self.MAX_TIME = 6000 # 60 seconds at 100 steps/sec equivalent

        # --- Colors ---
        self.COLOR_BG = (25, 20, 40)
        self.COLOR_GRID_BG = (35, 30, 55)
        self.COLOR_GRID_LINES = (50, 45, 70)
        self.COLOR_UI_TEXT = (240, 240, 255)
        self.COLOR_UI_BAR_BG = (60, 50, 80)
        self.COLOR_UI_BAR_FILL = (100, 220, 255)
        self.COLOR_CURSOR = (255, 255, 255)

        self.GEM_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (100, 150, 255), # Blue
            (255, 240, 100), # Yellow
            (255, 100, 255), # Magenta
            (100, 255, 255), # Cyan
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- Grid & Rendering Calculation ---
        self.GRID_MARGIN_X = 40
        self.GRID_MARGIN_Y = 60
        self.GRID_AREA_WIDTH = self.WIDTH - 2 * self.GRID_MARGIN_X
        self.GRID_AREA_HEIGHT = self.HEIGHT - self.GRID_MARGIN_Y - 20
        self.CELL_W = self.GRID_AREA_WIDTH / self.GRID_SIZE[0]
        self.CELL_H = self.GRID_AREA_HEIGHT / self.GRID_SIZE[1]
        self.GEM_SIZE = min(self.CELL_W, self.CELL_H) * 0.4

        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.timer = None
        self.collected_gems_total = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.last_action_was_collect = False
        
        self.reset()
        
        # --- Final Validation ---
        self.validate_implementation()

    def _generate_grid(self, fill=True):
        if fill:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=self.GRID_SIZE[::-1])
        
        # Ensure no initial matches
        while True:
            matches_found = False
            for r in range(self.GRID_SIZE[1]):
                for c in range(self.GRID_SIZE[0]):
                    if len(self._find_matches(c, r)) >= self.MATCH_MIN:
                        matches_found = True
                        self.grid[r, c] = (self.grid[r, c] + 1) % self.NUM_GEM_TYPES
            if not matches_found:
                break
    
    def _check_for_valid_moves(self):
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                if len(self._find_matches(c, r)) >= self.MATCH_MIN:
                    return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.score = 0
        self.timer = self.MAX_TIME
        self.collected_gems_total = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.last_action_was_collect = False

        self._generate_grid()
        if not self._check_for_valid_moves():
            self._generate_grid() # Regenerate if stuck

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.last_action_was_collect = False

        # --- Action: Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_SIZE[0]
        self.cursor_pos[1] %= self.GRID_SIZE[1]

        # --- Action: Select Gem (Space) ---
        if space_held:
            cx, cy = self.cursor_pos
            if self.grid[cy, cx] != -1: # Cannot select an empty space
                matches = self._find_matches(cx, cy)
                num_matches = len(matches)

                if num_matches >= self.MATCH_MIN:
                    # Successful collection
                    # SFX: gem_collect_success.wav
                    self.last_action_was_collect = True
                    gem_type = self.grid[cy, cx]
                    
                    for r, c in matches:
                        self.grid[r, c] = -1 # Mark as empty
                        self._create_particles((c, r), self.GEM_COLORS[gem_type])

                    reward += num_matches  # +1 per gem
                    if num_matches >= 4:
                        reward += 2 # Bonus for larger match

                    self.score += int(reward * 10)
                    self.collected_gems_total += num_matches
                    
                    self._apply_gravity()
                    self._fill_new_gems()

                    if not self._check_for_valid_moves():
                        self._generate_grid() # Reshuffle if no moves left
                        # SFX: board_reshuffle.wav

                elif num_matches == 2:
                    # Identified a potential pair
                    reward += 0.1
                else:
                    # Selected a lone gem
                    reward -= 0.1
                    # SFX: gem_select_fail.wav

        # --- Update State ---
        self.steps += 1
        self.timer -= 1
        self._update_particles()

        # --- Check Termination ---
        terminated = False
        if self.collected_gems_total >= self.WIN_GEMS_TARGET:
            reward += 100
            terminated = True
            # SFX: game_win.wav
        elif self.timer <= 0:
            reward -= 100
            terminated = True
            # SFX: game_over.wav
        
        self.game_over = terminated
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _find_matches(self, start_c, start_r):
        if not (0 <= start_c < self.GRID_SIZE[0] and 0 <= start_r < self.GRID_SIZE[1]):
            return []
        
        target_gem_type = self.grid[start_r, start_c]
        if target_gem_type == -1:
            return []

        q = deque([(start_r, start_c)])
        visited = set([(start_r, start_c)])
        matches = []

        while q:
            r, c = q.popleft()
            matches.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_SIZE[1] and 0 <= nc < self.GRID_SIZE[0]:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_gem_type:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_SIZE[0]):
            empty_slots = []
            for r in range(self.GRID_SIZE[1] - 1, -1, -1):
                if self.grid[r, c] == -1:
                    empty_slots.append(r)
                elif empty_slots:
                    new_r = empty_slots.pop(0)
                    self.grid[new_r, c] = self.grid[r, c]
                    self.grid[r, c] = -1
                    empty_slots.append(r)
                    empty_slots.sort(reverse=True)

    def _fill_new_gems(self):
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _create_particles(self, grid_pos, color, count=15):
        c, r = grid_pos
        px = self.GRID_MARGIN_X + (c + 0.5) * self.CELL_W
        py = self.GRID_MARGIN_Y + (r + 0.5) * self.CELL_H

        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_MARGIN_X, self.GRID_MARGIN_Y, self.GRID_AREA_WIDTH, self.GRID_AREA_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        # Draw gems
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                gem_type = self.grid[r, c]
                if gem_type != -1:
                    px = self.GRID_MARGIN_X + (c + 0.5) * self.CELL_W
                    py = self.GRID_MARGIN_Y + (r + 0.5) * self.CELL_H
                    self._draw_gem(px, py, self.GEM_COLORS[gem_type])

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_px = self.GRID_MARGIN_X + (cx + 0.5) * self.CELL_W
        cursor_py = self.GRID_MARGIN_Y + (cy + 0.5) * self.CELL_H
        
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        size = self.CELL_W * 0.5 + pulse * 4
        alpha = 150 + pulse * 105
        
        # Draw a soft pulsing glow
        cursor_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        pygame.draw.circle(cursor_surf, (*self.COLOR_CURSOR, int(alpha/3)), (size, size), size)
        pygame.draw.circle(cursor_surf, (*self.COLOR_CURSOR, int(alpha/2)), (size, size), size * 0.8)
        self.screen.blit(cursor_surf, (cursor_px - size, cursor_py - size), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw a sharp frame
        rect = pygame.Rect(cursor_px - self.CELL_W/2, cursor_py - self.CELL_H/2, self.CELL_W, self.CELL_H)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=4)
        
        # Draw particles
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = life_ratio * 3
            color = tuple(int(c * life_ratio) for c in p['color'])
            pygame.draw.circle(self.screen, color, p['pos'], radius)

    def _draw_gem(self, x, y, color):
        s = self.GEM_SIZE
        highlight = tuple(min(255, c + 60) for c in color)
        shadow = tuple(max(0, c - 60) for c in color)
        
        points = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # 3D effect
        pygame.draw.line(self.screen, highlight, (x, y-s), (x-s, y), 2)
        pygame.draw.line(self.screen, shadow, (x, y+s), (x+s, y), 2)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_main.render(f"SCORE: {self.score:06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 15, 15))

        # --- Timer ---
        time_sec = math.ceil(max(0, self.timer) / (self.MAX_TIME / 60))
        time_color = (255, 100, 100) if time_sec <= 10 else self.COLOR_UI_TEXT
        time_text = self.font_main.render(f"TIME: {time_sec:02d}", True, time_color)
        self.screen.blit(time_text, (self.WIDTH // 2 - time_text.get_width() // 2, 15))

        # --- Gem Collection Progress Bar ---
        bar_x, bar_y, bar_w, bar_h = 15, 20, 200, 20
        
        # Bar background
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        
        # Bar fill
        fill_ratio = min(1.0, self.collected_gems_total / self.WIN_GEMS_TARGET)
        if fill_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (bar_x, bar_y, bar_w * fill_ratio, bar_h), border_radius=5)
        
        # Bar text
        gem_text = self.font_small.render(f"GEMS: {self.collected_gems_total}/{self.WIN_GEMS_TARGET}", True, self.COLOR_UI_TEXT)
        self.screen.blit(gem_text, (bar_x + bar_w // 2 - gem_text.get_width() // 2, bar_y + bar_h // 2 - gem_text.get_height() // 2))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.collected_gems_total >= self.WIN_GEMS_TARGET:
                end_text_str = "YOU WIN!"
                end_color = (150, 255, 150)
            else:
                end_text_str = "TIME'S UP!"
                end_color = (255, 150, 150)
                
            end_text = self.font_main.render(end_text_str, True, end_color)
            self.screen.blit(end_text, (self.WIDTH//2 - end_text.get_width()//2, self.HEIGHT//2 - end_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "collected_gems": self.collected_gems_total,
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    # Game loop
    while running:
        # Default action is no-op
        action = [0, 0, 0] # move=none, space=released, shift=released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        # Get key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # The game state will be frozen, press 'R' to reset.

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Wait for a short duration to make the game playable by humans
        pygame.time.wait(30)

    env.close()