
# Generated: 2025-08-27T21:37:40.236564
# Source Brief: brief_02850.md
# Brief Index: 2850

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from itertools import combinations
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to rake the sand. Press Shift to cycle through rock positions, and Space to place a rock."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A serene Zen garden simulation. Rake patterns in the sand and strategically place rocks to maximize your aesthetic score before the time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_TIME = 60  # seconds
        self.WIN_SCORE = 80
        self.MAX_ROCKS = 5
        self.RAKE_TRAIL_DURATION = int(self.FPS * 10) # 10 seconds

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
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # Colors
        self.COLOR_SAND = (210, 180, 140)
        self.COLOR_SAND_RAKED = (188, 158, 125)
        self.COLOR_WALL = (139, 69, 19)
        self.COLOR_ROCK = (80, 80, 80)
        self.COLOR_ROCK_SHADOW = (60, 60, 60)
        self.COLOR_UI_TEXT = (50, 50, 50)
        self.COLOR_AESTHETIC_BAR = (102, 194, 165)
        self.COLOR_TIMER_BAR_GOOD = (95, 158, 160)
        self.COLOR_TIMER_BAR_BAD = (220, 20, 60)
        self.COLOR_UI_BAR_BG = (200, 200, 200)
        self.COLOR_CURSOR = (144, 238, 144)

        # Game state variables
        self.rng = None
        self.timer = 0
        self.aesthetic_score = 0
        self.rocks_placed = []
        self.rocks_remaining = 0
        self.game_over = False
        self.steps = 0
        self.last_space_press = False
        self.last_shift_press = False

        # Rake and Sand Grid
        self.RAKE_GRID_SCALE = 10
        self.rake_grid_w = self.WIDTH // self.RAKE_GRID_SCALE
        self.rake_grid_h = self.HEIGHT // self.RAKE_GRID_SCALE
        self.sand_grid = np.zeros((self.rake_grid_h, self.rake_grid_w), dtype=np.int32)
        self.rake_pos = (0, 0)

        # Rock Placement Grid
        self._define_placement_locations()
        self.placement_cursor_idx = 0
        
        # Initialize state
        self.reset()
        self.validate_implementation()

    def _define_placement_locations(self):
        self.valid_placement_locs = []
        margin_x, margin_y = 80, 60
        rows, cols = 3, 4
        for r in range(rows):
            for c in range(cols):
                x = margin_x + c * (self.WIDTH - 2 * margin_x) / (cols - 1)
                y = margin_y + r * (self.HEIGHT - 2 * margin_y) / (rows - 1)
                self.valid_placement_locs.append((int(x), int(y)))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.steps = 0
        self.timer = self.MAX_TIME
        self.aesthetic_score = 0
        self.game_over = False
        self.rocks_placed = []
        self.rocks_remaining = self.MAX_ROCKS
        self.last_space_press = False
        self.last_shift_press = False
        self.sand_grid.fill(0)
        self.rake_pos = (self.rake_grid_w // 2, self.rake_grid_h // 2)
        self.placement_cursor_idx = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self.timer -= 1 / self.FPS
        self.steps += 1
        
        # Decrement sand trail lifetimes
        self.sand_grid = np.maximum(0, self.sand_grid - 1)

        # --- Handle Actions ---
        prev_aesthetic_score = self._calculate_aesthetics()

        # 1. Raking (Movement)
        old_rake_pos = self.rake_pos
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right

        if dx != 0 or dy != 0:
            new_rake_x = np.clip(self.rake_pos[0] + dx, 2, self.rake_grid_w - 3)
            new_rake_y = np.clip(self.rake_pos[1] + dy, 2, self.rake_grid_h - 3)
            self.rake_pos = (new_rake_x, new_rake_y)
            
            # Draw line on sand grid
            is_redundant = self.sand_grid[new_rake_y, new_rake_x] > 0
            self._rake_line_on_grid(old_rake_pos, self.rake_pos)
            
            # Raking reward
            if is_redundant:
                reward -= 0.2 # Penalize raking over existing trails
            else:
                reward += 0.1 # Reward for new trails
            # sfx: sand_rake.wav

        # 2. Cycle placement cursor (Shift)
        if shift_held and not self.last_shift_press:
            self.placement_cursor_idx = (self.placement_cursor_idx + 1) % len(self.valid_placement_locs)
            # sfx: ui_tick.wav
        self.last_shift_press = shift_held

        # 3. Place Rock (Space)
        if space_held and not self.last_space_press and self.rocks_remaining > 0:
            pos_to_place = self.valid_placement_locs[self.placement_cursor_idx]
            
            # Check if a rock is already there
            can_place = True
            for rock in self.rocks_placed:
                if np.linalg.norm(np.array(rock['pos']) - np.array(pos_to_place)) < rock['radius'] * 2:
                    can_place = False
                    break
            
            if can_place:
                rock_size = self.rng.integers(20, 35)
                self.rocks_placed.append({'pos': pos_to_place, 'radius': rock_size})
                self.rocks_remaining -= 1
                # sfx: rock_place.wav

                # Rock placement reward
                new_aesthetic_score = self._calculate_aesthetics()
                if new_aesthetic_score > prev_aesthetic_score:
                    reward += 5
                else:
                    reward -= 1

        self.last_space_press = space_held

        # --- Update Score and Check Termination ---
        self.aesthetic_score = self._calculate_aesthetics()
        terminated = False

        if self.aesthetic_score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
            # sfx: win_chime.wav
        
        if self.timer <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
            # sfx: lose_buzz.wav

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _rake_line_on_grid(self, p1, p2):
        """Bresenham's line algorithm on the sand grid."""
        x1, y1 = p1
        x2, y2 = p2
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            self.sand_grid[y1, x1] = self.RAKE_TRAIL_DURATION
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    def _calculate_aesthetics(self):
        """Calculates the total aesthetic score based on rocks and sand."""
        sand_score = self._calculate_sand_score() # Max 40
        rock_score = self._calculate_rock_score() # Max 60
        return np.clip(sand_score + rock_score, 0, 100)

    def _calculate_sand_score(self):
        """Scores the sand patterns. Max 40 points."""
        raked_cells = np.count_nonzero(self.sand_grid)
        total_cells = self.rake_grid_w * self.rake_grid_h
        
        # Reward for covering a decent area, with diminishing returns
        coverage_ratio = raked_cells / total_cells
        score = 40 * (1 - math.exp(-5 * coverage_ratio)) # Exponential curve for nice falloff
        return score

    def _calculate_rock_score(self):
        """Scores the rock arrangement. Max 60 points."""
        if len(self.rocks_placed) < 2:
            return 0

        score = 0
        
        # 1. Spacing (up to 20 pts)
        min_dist = float('inf')
        total_dist = 0
        for r1, r2 in combinations(self.rocks_placed, 2):
            dist = np.linalg.norm(np.array(r1['pos']) - np.array(r2['pos']))
            total_dist += dist
            if dist < min_dist:
                min_dist = dist
        
        # Penalize rocks being too close
        if min_dist < 50:
            score -= (50 - min_dist) * 0.5
        
        # Reward good average spacing
        avg_dist = total_dist / len(list(combinations(self.rocks_placed, 2)))
        spacing_score = np.clip(avg_dist / 20, 0, 20) # Normalize
        score += spacing_score

        # 2. Asymmetry and Balance (up to 20 pts)
        if len(self.rocks_placed) > 0:
            rock_positions = np.array([r['pos'] for r in self.rocks_placed])
            center_of_mass = np.mean(rock_positions, axis=0)
            garden_center = np.array([self.WIDTH / 2, self.HEIGHT / 2])
            
            # Reward for being off-center (asymmetry) but not too much
            offset_dist = np.linalg.norm(center_of_mass - garden_center)
            asymmetry_score = 20 * (1 - math.exp(-0.01 * offset_dist))
            score += asymmetry_score

        # 3. Composition (triangles) (up to 20 pts)
        if len(self.rocks_placed) >= 3:
            total_area = 0
            for r1, r2, r3 in combinations(self.rocks_placed, 3):
                p1, p2, p3 = np.array(r1['pos']), np.array(r2['pos']), np.array(r3['pos'])
                # Shoelace formula for triangle area
                area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                total_area += area
            
            # Reward larger, non-degenerate triangles
            composition_score = np.clip(total_area / 10000, 0, 20)
            score += composition_score

        return np.clip(score, 0, 60)

    def _get_observation(self):
        # Clear screen with wall color
        self.screen.fill(self.COLOR_WALL)
        
        # Draw sand area
        sand_rect = pygame.Rect(10, 10, self.WIDTH - 20, self.HEIGHT - 20)
        pygame.draw.rect(self.screen, self.COLOR_SAND, sand_rect)
        
        # Render game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render sand trails
        for y in range(self.rake_grid_h):
            for x in range(self.rake_grid_w):
                if self.sand_grid[y, x] > 0:
                    px = x * self.RAKE_GRID_SCALE + self.RAKE_GRID_SCALE // 2
                    py = y * self.RAKE_GRID_SCALE + self.RAKE_GRID_SCALE // 2
                    alpha = int(np.clip(self.sand_grid[y, x] / (self.RAKE_TRAIL_DURATION / 2), 0, 1) * 100)
                    color = (*self.COLOR_SAND_RAKED, alpha)
                    # Use a surface to draw with alpha
                    temp_surf = pygame.Surface((self.RAKE_GRID_SCALE, self.RAKE_GRID_SCALE), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (self.RAKE_GRID_SCALE//2, self.RAKE_GRID_SCALE//2), self.RAKE_GRID_SCALE//2)
                    self.screen.blit(temp_surf, (px - self.RAKE_GRID_SCALE//2, py - self.RAKE_GRID_SCALE//2))

        # Render rock placement cursor
        if self.rocks_remaining > 0 and not self.game_over:
            cursor_pos = self.valid_placement_locs[self.placement_cursor_idx]
            pulse = abs(math.sin(self.steps * 0.1))
            radius = int(25 + pulse * 5)
            # Use a surface for transparency
            cursor_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(cursor_surf, radius, radius, radius, (*self.COLOR_CURSOR, 100))
            pygame.gfxdraw.aacircle(cursor_surf, radius, radius, radius, (*self.COLOR_CURSOR, 200))
            self.screen.blit(cursor_surf, (cursor_pos[0]-radius, cursor_pos[1]-radius))

        # Render placed rocks
        for rock in self.rocks_placed:
            pos = rock['pos']
            radius = rock['radius']
            # Shadow
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0] + 3), int(pos[1] + 3), radius, self.COLOR_ROCK_SHADOW)
            # Rock
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ROCK)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), radius, self.COLOR_ROCK)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Aesthetic: {int(self.aesthetic_score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Aesthetic Score Bar
        bar_width = 150
        bar_height = 10
        score_ratio = np.clip(self.aesthetic_score / self.WIN_SCORE, 0, 1)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (20, 50, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_AESTHETIC_BAR, (20, 50, bar_width * score_ratio, bar_height))

        # Timer
        time_ratio = np.clip(self.timer / self.MAX_TIME, 0, 1)
        timer_color = self.COLOR_TIMER_BAR_GOOD if time_ratio > 0.25 else self.COLOR_TIMER_BAR_BAD
        timer_bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (self.WIDTH - timer_bar_width - 20, 20, timer_bar_width, bar_height))
        pygame.draw.rect(self.screen, timer_color, (self.WIDTH - timer_bar_width - 20, 20, timer_bar_width * time_ratio, bar_height))

        # Rock Counter
        rock_text = self.font_small.render(f"Rocks: {self.rocks_remaining}/{self.MAX_ROCKS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(rock_text, (20, self.HEIGHT - 30))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            message = "Garden Complete" if self.aesthetic_score >= self.WIN_SCORE else "Time's Up"
            end_text = self.font_main.render(message, True, (255, 255, 255))
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.aesthetic_score,
            "steps": self.steps,
            "timer": self.timer,
            "rocks_remaining": self.rocks_remaining,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Zen Garden")
    
    done = False
    clock = pygame.time.Clock()
    
    # --- Instructions ---
    print("\n" + "="*30)
    print("Zen Garden - Player Mode")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Keyboard Controls to MultiDiscrete Action ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # no-op
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space button
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift button
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.2f}")

        clock.tick(env.FPS)
        
    print("\nGame Over!")
    print(f"Final Score: {info['score']:.1f}")
    print(f"Total Steps: {info['steps']}")
    
    env.close()