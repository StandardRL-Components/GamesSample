import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Press Space to plant or harvest. "
        "Go to the market (red stall) and press Shift to sell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage your farm by planting, harvesting, and selling crops. Earn 1000 coins before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    WIN_SCORE = 1000
    MAX_STEPS = 300 * 5  # 5 minutes total steps.
    PLAYER_SPEED = 20
    GROWTH_TIME = 50
    CROP_SELL_VALUE = 25

    # --- Rewards ---
    REWARD_PLANT = 0.1
    REWARD_HARVEST = 0.2
    REWARD_SELL_BASE = 1.0  # This gets multiplied by number of crops sold
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    # --- Colors ---
    COLOR_BG = (45, 35, 25)
    COLOR_PLOT_EMPTY = (102, 72, 54)
    COLOR_PLOT_GROWING_START_TUPLE = (60, 140, 60)
    COLOR_PLOT_GROWING_END_TUPLE = (80, 220, 80)
    COLOR_PLOT_READY = (240, 220, 50)
    COLOR_MARKET = (200, 40, 40)
    COLOR_PLAYER = (230, 230, 250)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TIMER_BAR = (50, 100, 200)
    COLOR_TIMER_BAR_WARN = (200, 100, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # Convert tuples to pygame.Color for lerp functionality
        self.COLOR_PLOT_GROWING_START = pygame.Color(*self.COLOR_PLOT_GROWING_START_TUPLE)
        self.COLOR_PLOT_GROWING_END = pygame.Color(*self.COLOR_PLOT_GROWING_END_TUPLE)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.plots = []
        self.market_rect = None
        self.player_inventory = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []

        # Initialize state variables
        # self.reset() is called by the validation method

        # Run validation check
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_inventory = 0
        self.particles = []

        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2]

        self.market_rect = pygame.Rect(self.SCREEN_WIDTH - 60, self.SCREEN_HEIGHT // 2 - 50, 50, 100)

        self.plots = []
        plot_size = 70
        spacing = 20
        grid_w, grid_h = 4, 3
        total_grid_w = grid_w * plot_size + (grid_w - 1) * spacing
        total_grid_h = grid_h * plot_size + (grid_h - 1) * spacing
        start_x = (self.SCREEN_WIDTH - total_grid_w - 80) // 2
        start_y = (self.SCREEN_HEIGHT - total_grid_h) // 2

        for row in range(grid_h):
            for col in range(grid_w):
                x = start_x + col * (plot_size + spacing)
                y = start_y + row * (plot_size + spacing)
                self.plots.append({
                    "rect": pygame.Rect(x, y, plot_size, plot_size),
                    "state": "empty",  # empty, growing, ready
                    "growth": 0.0
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0

        # Unpack factorized action
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- 1. Update Player Position ---
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)

        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 10, 20, 20)

        # --- 2. Handle Actions (Space & Shift) ---
        if space_pressed:
            for plot in self.plots:
                if player_rect.colliderect(plot["rect"]):
                    if plot["state"] == "empty":
                        # Plant
                        plot["state"] = "growing"
                        plot["growth"] = 0.0
                        reward += self.REWARD_PLANT
                        self._create_particles(plot["rect"].center, self.COLOR_PLOT_GROWING_END_TUPLE, 10)
                        break
                    elif plot["state"] == "ready":
                        # Harvest
                        plot["state"] = "empty"
                        plot["growth"] = 0.0
                        self.player_inventory += 1
                        reward += self.REWARD_HARVEST
                        self._create_particles(plot["rect"].center, self.COLOR_PLOT_READY, 20)
                        break

        if shift_pressed:
            if player_rect.colliderect(self.market_rect) and self.player_inventory > 0:
                # Sell
                coins_earned = self.player_inventory * self.CROP_SELL_VALUE
                reward += self.REWARD_SELL_BASE * self.player_inventory
                self.score += coins_earned
                self._create_particles(self.player_pos, (255, 215, 0), self.player_inventory * 5, is_coin=True)
                self.player_inventory = 0

        # --- 3. Update Game State ---
        self.steps += 1

        for plot in self.plots:
            if plot["state"] == "growing":
                plot["growth"] += 1.0 / self.GROWTH_TIME
                if plot["growth"] >= 1.0:
                    plot["state"] = "ready"
                    plot["growth"] = 1.0

        # --- 4. Check Termination ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += self.REWARD_LOSE
            terminated = True
            self.game_over = True

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
        self._update_and_render_particles()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Render Market
        pygame.draw.rect(self.screen, self.COLOR_MARKET, self.market_rect, border_radius=8)
        market_text = self.font_small.render("SELL", True, self.COLOR_TEXT)
        self.screen.blit(market_text, market_text.get_rect(center=self.market_rect.center))

        # Render Plots
        for plot in self.plots:
            pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, plot["rect"], border_radius=5)
            if plot["state"] == "growing":
                progress = plot["growth"]
                color = self.COLOR_PLOT_GROWING_START.lerp(self.COLOR_PLOT_GROWING_END, progress)
                # Draw a growing circle inside the plot
                radius = int((plot["rect"].width / 2 - 5) * progress)
                pygame.draw.circle(self.screen, color, plot["rect"].center, max(0, radius))
            elif plot["state"] == "ready":
                # Draw a filled circle to indicate readiness
                pygame.draw.circle(self.screen, self.COLOR_PLOT_READY, plot["rect"].center, plot["rect"].width / 2 - 5)
                # Add a little shine effect
                shine_rect = pygame.Rect(0, 0, 10, 10)
                shine_rect.center = (plot["rect"].centerx + 10, plot["rect"].centery - 10)
                pygame.draw.ellipse(self.screen, (255, 255, 200), shine_rect)

        # Render Player
        player_x, player_y = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, player_x, player_y, 10, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_x, player_y, 10, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"COINS: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Inventory
        inventory_text = self.font_small.render(f"Crops: {self.player_inventory}", True, self.COLOR_TEXT)
        self.screen.blit(inventory_text, (10, 40))

        # Timer Bar
        time_ratio = 1.0 - (self.steps / self.MAX_STEPS)
        bar_width = 200
        bar_height = 20
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10

        # Background of the bar
        pygame.draw.rect(self.screen, (30, 30, 30), (bar_x, bar_y, bar_width, bar_height), border_radius=5)

        # Foreground of the bar
        current_width = int(bar_width * time_ratio)
        bar_color = self.COLOR_TIMER_BAR if time_ratio > 0.25 else self.COLOR_TIMER_BAR_WARN
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, current_width, bar_height), border_radius=5)

        # Timer text
        timer_text = self.font_small.render("TIME", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (bar_x + bar_width / 2 - timer_text.get_width() / 2, bar_y + bar_height / 2 - timer_text.get_height() / 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "inventory": self.player_inventory,
        }

    def _create_particles(self, pos, color, count, is_coin=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "lifetime": lifetime,
                "max_lifetime": lifetime,
                "color": color,
                "is_coin": is_coin
            })

    def _update_and_render_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifetime"] -= 1

            if p["lifetime"] > 0:
                active_particles.append(p)

                alpha = int(255 * (p["lifetime"] / p["max_lifetime"]))
                color_with_alpha = (*p["color"], alpha)

                size = 8 if p["is_coin"] else 4
                particle_surf = pygame.Surface((size, size), pygame.SRCALPHA)

                if p["is_coin"]:
                    pygame.draw.circle(particle_surf, color_with_alpha, (size // 2, size // 2), size // 2)
                    pygame.draw.circle(particle_surf, (255, 255, 100, alpha), (size // 2, size // 2), size // 4)
                else:
                    particle_surf.fill(color_with_alpha)

                self.screen.blit(particle_surf, (int(p["pos"][0]), int(p["pos"][1])))

        self.particles = active_particles

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

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {obs.shape}"
        assert obs.dtype == np.uint8, f"Obs dtype is {obs.dtype}"
        assert isinstance(info, dict)

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # We need to unset the headless mode for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    env.validate_implementation()
    obs, info = env.reset()
    done = False

    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Farming Sim")
    clock = pygame.time.Clock()

    print(env.game_description)
    print(env.user_guide)

    total_reward = 0

    while not done:
        # --- Action mapping for human input ---
        keys = pygame.key.get_pressed()
        movement = 0  # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_pressed, shift_pressed]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # --- Frame rate control ---
        clock.tick(10)  # 10 steps per second for a responsive feel

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()