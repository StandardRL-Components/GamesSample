
# Generated: 2025-08-27T15:42:48.073925
# Source Brief: brief_01056.md
# Brief Index: 1056

        
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
        "Controls: Use arrow keys to select a plot. Press Space to plant or harvest. Press Shift to sell all harvested crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced farming game. Plant seeds, harvest crops, and sell them at the market to earn 500 coins before time runs out."
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
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_ROWS, self.GRID_COLS = 3, 5
        self.PLOT_SIZE = 70
        self.PLOT_SPACING = 15
        self.GRID_WIDTH = self.GRID_COLS * (self.PLOT_SIZE + self.PLOT_SPACING) - self.PLOT_SPACING
        self.GRID_HEIGHT = self.GRID_ROWS * (self.PLOT_SIZE + self.PLOT_SPACING) - self.PLOT_SPACING
        self.GRID_X_START = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_START = 100

        self.MAX_STEPS = 600  # 60 seconds at 10 steps/sec
        self.WIN_COINS = 500
        self.CROP_GROW_TIME = 50  # steps to grow
        self.CROP_SELL_PRICE = 5

        # Colors
        self.COLOR_BG = (40, 60, 40)
        self.COLOR_PLOT_EMPTY = (139, 90, 43)
        self.COLOR_PLOT_PLANTED = (34, 139, 34)
        self.COLOR_PLOT_GROWN = (124, 252, 0)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_MARKET = (190, 50, 50)
        self.COLOR_HARVESTED_CROP = (255, 223, 0)
        self.COLOR_GOLD_PARTICLE = (255, 215, 0)

        # Fonts
        self.font_large = pygame.font.Font(None, 60)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # State variables (initialized in reset)
        self.steps = 0
        self.time_left = 0
        self.coins = 0
        self.harvested_crops = 0
        self.plots = []
        self.selected_plot = (0, 0)
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.time_left = self.MAX_STEPS
        self.coins = 0
        self.harvested_crops = 0
        self.game_over = False
        self.win_message = ""
        self.particles = []

        self.plots = [
            [{'state': 'empty', 'growth': 0} for _ in range(self.GRID_COLS)]
            for _ in range(self.GRID_ROWS)
        ]
        
        self.selected_plot = (self.GRID_ROWS // 2, self.GRID_COLS // 2)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        reward = 0.0

        # --- Update Crop Growth ---
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot = self.plots[r][c]
                if plot['state'] == 'planted':
                    plot['growth'] += 1
                    if plot['growth'] >= self.CROP_GROW_TIME:
                        plot['state'] = 'grown'
        
        # --- Process Actions ---
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        
        r, c = self.selected_plot
        if movement == 1: r = max(0, r - 1)
        elif movement == 2: r = min(self.GRID_ROWS - 1, r + 1)
        elif movement == 3: c = max(0, c - 1)
        elif movement == 4: c = min(self.GRID_COLS - 1, c + 1)
        self.selected_plot = (r, c)

        if space_press:
            plot = self.plots[r][c]
            plot_center = self._get_plot_center(r, c)
            if plot['state'] == 'empty':
                plot['state'] = 'planted'
                plot['growth'] = 0
                # sfx: plant_seed.wav
                self._create_particles(plot_center, self.COLOR_PLOT_PLANTED, 5, speed_range=(0.5, 1.5))
            elif plot['state'] == 'grown':
                plot['state'] = 'empty'
                plot['growth'] = 0
                self.harvested_crops += 1
                reward += 0.1
                # sfx: harvest.wav
                self._create_particles(plot_center, self.COLOR_HARVESTED_CROP, 10)
            else:
                reward -= 0.01 # Penalty for failed action

        if shift_press:
            if self.harvested_crops > 0:
                sold_count = self.harvested_crops
                self.coins += sold_count * self.CROP_SELL_PRICE
                self.harvested_crops = 0
                reward += 1.0 * sold_count
                # sfx: cash_register.wav
                market_center = (self.WIDTH - 70, self.HEIGHT // 2)
                self._create_particles(market_center, self.COLOR_GOLD_PARTICLE, 20 + sold_count)
            else:
                reward -= 0.01 # Penalty for failed action

        # --- Check Termination ---
        terminated = False
        if self.coins >= self.WIN_COINS:
            terminated = True
            self.game_over = True
            reward += 100
            self.win_message = "YOU WIN!"
        elif self.time_left <= 0:
            terminated = True
            self.game_over = True
            reward -= 100
            self.win_message = "TIME'S UP!"

        self._update_particles()
            
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.coins, "steps": self.steps}

    def _get_plot_center(self, r, c):
        x = self.GRID_X_START + c * (self.PLOT_SIZE + self.PLOT_SPACING) + self.PLOT_SIZE // 2
        y = self.GRID_Y_START + r * (self.PLOT_SIZE + self.PLOT_SPACING) + self.PLOT_SIZE // 2
        return (x, y)

    def _create_particles(self, pos, color, count, speed_range=(1, 4)):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(*speed_range)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': 20, 'color': color})

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1

    def _render_game(self):
        # Draw plots
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot_rect = pygame.Rect(
                    self.GRID_X_START + c * (self.PLOT_SIZE + self.PLOT_SPACING),
                    self.GRID_Y_START + r * (self.PLOT_SIZE + self.PLOT_SPACING),
                    self.PLOT_SIZE, self.PLOT_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, plot_rect, border_radius=8)

                plot = self.plots[r][c]
                if plot['state'] == 'planted':
                    ratio = plot['growth'] / self.CROP_GROW_TIME
                    color = tuple(int(s + (e - s) * ratio) for s, e in zip(self.COLOR_PLOT_PLANTED, self.COLOR_PLOT_GROWN))
                    size = int(self.PLOT_SIZE * 0.8 * ratio)
                    center = plot_rect.center
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], size // 2, color)
                elif plot['state'] == 'grown':
                    center = plot_rect.center
                    pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], int(self.PLOT_SIZE * 0.4), self.COLOR_PLOT_GROWN)
                    pygame.gfxdraw.aacircle(self.screen, center[0], center[1], int(self.PLOT_SIZE * 0.4), self.COLOR_BG)

        # Draw market stall
        market_rect = pygame.Rect(self.WIDTH - 120, self.HEIGHT // 2 - 60, 100, 120)
        pygame.draw.rect(self.screen, self.COLOR_MARKET, market_rect, border_radius=10)
        market_text = self.font_small.render("SELL", True, self.COLOR_TEXT)
        self.screen.blit(market_text, market_text.get_rect(center=market_rect.center))

        # Draw harvested crops display
        for i in range(min(self.harvested_crops, 24)):
            x = market_rect.left - 25 - (i % 3) * 15
            y = market_rect.top + 10 + (i // 3) * 15
            pygame.gfxdraw.filled_circle(self.screen, x, y, 6, self.COLOR_HARVESTED_CROP)
            pygame.gfxdraw.aacircle(self.screen, x, y, 6, self.COLOR_BG)

        # Draw selector
        r, c = self.selected_plot
        selector_rect = pygame.Rect(
            self.GRID_X_START + c * (self.PLOT_SIZE + self.PLOT_SPACING) - 5,
            self.GRID_Y_START + r * (self.PLOT_SIZE + self.PLOT_SPACING) - 5,
            self.PLOT_SIZE + 10, self.PLOT_SIZE + 10)
        alpha = 128 + 127 * math.sin(self.steps * 0.3)
        s = pygame.Surface(selector_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(s, (*self.COLOR_SELECTOR, alpha), s.get_rect(), width=4, border_radius=12)
        self.screen.blit(s, selector_rect.topleft)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = (*p['color'], alpha)
            size = max(1, int(p['life'] / 4))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)

    def _render_ui(self):
        # UI Background panels
        pygame.draw.rect(self.screen, (0,0,0,100), (0, 0, self.WIDTH, 80))

        # Coins display
        coins_text = self.font_medium.render(f"Coins: {self.coins} / {self.WIN_COINS}", True, self.COLOR_TEXT)
        self.screen.blit(coins_text, (20, 15))

        # Time display
        time_str = f"Time: {self.time_left / 10.0:.1f}"
        time_text = self.font_medium.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 15))

        # Harvested crops display
        harvest_text = self.font_medium.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_TEXT)
        self.screen.blit(harvest_text, (20, 45))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            end_text = self.font_large.render(self.win_message, True, self.COLOR_TEXT)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you need a window.
    # This part is for testing and will not work in a headless environment.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Farm Manager")
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            # Action defaults
            movement, space, shift = 0, 0, 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            env.clock.tick(10) # Control the speed of manual play

    except Exception as e:
        print("\nCould not create Pygame display. This is expected in a headless environment.")
        print("Manual play is unavailable. The environment itself is still valid.")
        
    finally:
        pygame.quit()