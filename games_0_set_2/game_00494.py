
# Generated: 2025-08-27T13:49:13.975904
# Source Brief: brief_00494.md
# Brief Index: 494

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to collect the gem under the cursor."
    )

    game_description = (
        "Race against the clock to collect all the gems scattered on the grid. "
        "Red gems are worth 3 points, blue are worth 2, and green are worth 1. "
        "Plan your route to maximize your score before time runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.fps = 30

        # --- Colors ---
        self.COLOR_BG = (15, 19, 26)
        self.COLOR_GRID = (40, 45, 55)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.GEM_COLORS = {
            1: (0, 255, 120),    # Green
            2: (0, 180, 255),    # Blue
            3: (255, 80, 80),    # Red
        }

        # --- Fonts ---
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_title = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game Layout ---
        self.grid_size = (10, 10)
        self.grid_area_size = 360
        self.cell_size = self.grid_area_size // self.grid_size[0]
        self.grid_top_left = (
            (self.screen_width - self.grid_area_size) // 2,
            (self.screen_height - self.grid_area_size) // 2,
        )

        # --- Game State ---
        self.np_random = None
        self.cursor_pos = [0, 0]
        self.gems = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.time_limit = 0.0
        self.game_over = False
        self.game_won = False
        self.move_cooldown = 0
        self.move_cooldown_max = 3 # frames per move

        self.num_gems_total = 20
        self.initial_time = 60.0
        self.max_steps = int(self.initial_time * self.fps) + 1 # Safety margin

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.time_limit = self.initial_time
        self.cursor_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        self.particles = []
        self.move_cooldown = 0
        
        self._generate_gems()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update Time and Steps ---
        self.time_limit -= 1.0 / self.fps
        self.steps += 1
        
        # --- Unpack Actions ---
        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1

        dist_before = self._find_nearest_gem_dist()

        # --- Handle Movement ---
        self.move_cooldown = max(0, self.move_cooldown - 1)
        if self.move_cooldown == 0 and movement != 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.grid_size[0] - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.grid_size[1] - 1)
            self.move_cooldown = self.move_cooldown_max

        # --- Handle Collection ---
        if space_pressed:
            gem_collected = None
            for gem in self.gems:
                if gem['pos'] == self.cursor_pos:
                    gem_collected = gem
                    break
            
            if gem_collected:
                self.gems.remove(gem_collected)
                self.score += gem_collected['value']
                reward += gem_collected['value']
                # sfx: gem collect
                self._add_particles(self.cursor_pos, gem_collected['color'], 30)

        # --- Movement Reward ---
        if self.gems:
            dist_after = self._find_nearest_gem_dist()
            if dist_after < dist_before:
                reward += 0.1
            elif dist_after > dist_before:
                reward -= 0.01
        
        # --- Check Termination Conditions ---
        if not self.gems:
            self.game_over = True
            self.game_won = True
            terminated = True
            reward += 50
            # sfx: win fanfare
        elif self.time_limit <= 0 or self.steps >= self.max_steps:
            self.game_over = True
            self.game_won = False
            terminated = True
            reward -= 50
            # sfx: lose buzzer

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

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_limit,
            "gems_remaining": len(self.gems)
        }

    def _render_game(self):
        # Draw grid
        for i in range(self.grid_size[0] + 1):
            start_pos = (self.grid_top_left[0] + i * self.cell_size, self.grid_top_left[1])
            end_pos = (self.grid_top_left[0] + i * self.cell_size, self.grid_top_left[1] + self.grid_area_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for i in range(self.grid_size[1] + 1):
            start_pos = (self.grid_top_left[0], self.grid_top_left[1] + i * self.cell_size)
            end_pos = (self.grid_top_left[0] + self.grid_area_size, self.grid_top_left[1] + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw gems
        gem_radius = self.cell_size // 4
        for gem in self.gems:
            cx = self.grid_top_left[0] + int((gem['pos'][0] + 0.5) * self.cell_size)
            cy = self.grid_top_left[1] + int((gem['pos'][1] + 0.5) * self.cell_size)
            color = gem['color']
            
            # Glow effect
            glow_radius = int(gem_radius * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, (*color, 50))
            self.screen.blit(glow_surf, (cx - glow_radius, cy - glow_radius))
            
            # Main gem (diamond shape)
            points = [
                (cx, cy - gem_radius),
                (cx + gem_radius, cy),
                (cx, cy + gem_radius),
                (cx - gem_radius, cy),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Draw particles
        self._update_and_draw_particles()

        # Draw cursor
        cursor_x = self.grid_top_left[0] + self.cursor_pos[0] * self.cell_size
        cursor_y = self.grid_top_left[1] + self.cursor_pos[1] * self.cell_size
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.cell_size, self.cell_size)
        
        # Pulsing effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 100 + pulse * 100
        pygame.draw.rect(self.screen, (*self.COLOR_CURSOR, alpha), cursor_rect, 0, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

    def _render_ui(self):
        # Score
        self._draw_text(f"SCORE: {self.score}", self.font_main, self.COLOR_TEXT, (10, 10), align="topleft")
        # Timer
        time_str = f"TIME: {max(0, self.time_limit):.1f}"
        time_color = (255, 100, 100) if self.time_limit < 10 else self.COLOR_TEXT
        self._draw_text(time_str, self.font_main, time_color, (self.screen_width - 10, 10), align="topright")
        # Gems remaining
        gem_str = f"GEMS: {len(self.gems)} / {self.num_gems_total}"
        self._draw_text(gem_str, self.font_small, self.COLOR_TEXT, (self.screen_width // 2, self.screen_height - 15), align="center")

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.game_won else "TIME'S UP!"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            self._draw_text(message, self.font_title, color, (self.screen_width // 2, self.screen_height // 2), align="center")

    def _generate_gems(self):
        self.gems = []
        all_pos = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        self.np_random.shuffle(all_pos)
        
        gem_positions = all_pos[:self.num_gems_total]
        
        # 10 green (1pt), 6 blue (2pt), 4 red (3pt)
        gem_values = [1] * 10 + [2] * 6 + [3] * 4
        self.np_random.shuffle(gem_values)

        for i in range(self.num_gems_total):
            value = gem_values[i]
            self.gems.append({
                'pos': list(gem_positions[i]),
                'value': value,
                'color': self.GEM_COLORS[value],
            })

    def _find_nearest_gem_dist(self):
        if not self.gems:
            return float('inf')
        
        min_dist = float('inf')
        for gem in self.gems:
            dist = abs(self.cursor_pos[0] - gem['pos'][0]) + abs(self.cursor_pos[1] - gem['pos'][1])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _add_particles(self, grid_pos, color, amount):
        px = self.grid_top_left[0] + int((grid_pos[0] + 0.5) * self.cell_size)
        py = self.grid_top_left[1] + int((grid_pos[1] + 0.5) * self.cell_size)
        for _ in range(amount):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.uniform(15, 30), # frames
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifespan'] / 30))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]),
                    int(p['radius']), (*p['color'], alpha)
                )

    def _draw_text(self, text, font, color, pos, align="topleft"):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topright":
            text_rect.topright = pos
        else: # topleft
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 1, text_rect.y + 1))
        self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Create a window to display the game
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = np.array([movement, space, 0])

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.fps)

    env.close()