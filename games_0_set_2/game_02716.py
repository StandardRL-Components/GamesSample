
# Generated: 2025-08-27T21:13:17.054145
# Source Brief: brief_02716.md
# Brief Index: 2716

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to plant or harvest. Press Shift to sell crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage your farm by planting, harvesting, and selling crops to earn 1000 coins before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 6000
        self.WIN_SCORE = 1000
        self.COINS_PER_CROP = 5

        # Grid properties
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.PLOT_SIZE = 36
        self.GRID_WIDTH = self.GRID_COLS * self.PLOT_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.PLOT_SIZE
        self.GRID_OFFSET_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT - 20)

        # Growth properties
        self.GROWTH_TIME = 240  # 8 seconds at 30fps

        # Plot states
        self.STATE_EMPTY = 0
        self.STATE_PLANTED = 1
        self.STATE_GROWN = 2

        # Colors (Bright and Contrasting)
        self.COLOR_BG = (25, 20, 20)
        self.COLOR_SOIL_BG = (76, 61, 49)
        self.COLOR_SOIL_PLOT = (101, 81, 65)
        self.COLOR_SEED = (144, 238, 144)
        self.COLOR_PLANT = (34, 139, 34)
        self.COLOR_GROWN = (255, 215, 0)
        self.COLOR_CURSOR = (0, 255, 255, 150)
        self.COLOR_MARKET_RED = (200, 30, 30)
        self.COLOR_MARKET_WHITE = (240, 240, 240)
        self.COLOR_UI_TEXT = (255, 255, 220)
        self.COLOR_UI_SHADOW = (20, 20, 20)
        self.COLOR_TIMER_GOOD = (0, 200, 0)
        self.COLOR_TIMER_WARN = (255, 255, 0)
        self.COLOR_TIMER_BAD = (255, 0, 0)
        
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
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 48, bold=True)
        
        # Initialize state variables
        self.farm_plots = []
        self.cursor_pos = [0, 0]
        self.harvested_crops = 0
        self.score = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.action_feedback = []
        
        # Initialize state variables
        self.reset()

    def _get_plot(self, x, y):
        return self.farm_plots[y * self.GRID_COLS + x]

    def _set_plot(self, x, y, state, growth=0):
        self.farm_plots[y * self.GRID_COLS + x] = {"state": state, "growth": growth}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.harvested_crops = 0
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.prev_space_held = False
        self.prev_shift_held = False
        self.action_feedback = []

        self.farm_plots = [
            {"state": self.STATE_EMPTY, "growth": 0}
            for _ in range(self.GRID_COLS * self.GRID_ROWS)
        ]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Update game logic
        self.steps += 1
        self.time_remaining -= 1

        reward = -0.001 # Small penalty for time passing, encourages action
        
        # 1. Handle player actions
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        if space_press:
            plot = self._get_plot(self.cursor_pos[0], self.cursor_pos[1])
            if plot["state"] == self.STATE_EMPTY:
                self._set_plot(self.cursor_pos[0], self.cursor_pos[1], self.STATE_PLANTED)
                self._add_feedback("Plant!", self.cursor_pos, self.COLOR_SEED)
                # sfx: plant_seed.wav
            elif plot["state"] == self.STATE_GROWN:
                self._set_plot(self.cursor_pos[0], self.cursor_pos[1], self.STATE_EMPTY)
                self.harvested_crops += 1
                reward += 0.1
                self._add_feedback("+1 Crop", self.cursor_pos, self.COLOR_GROWN)
                # sfx: harvest.wav
        
        if shift_press and self.harvested_crops > 0:
            coins_earned = self.harvested_crops * self.COINS_PER_CROP
            self.score += coins_earned
            reward += 1.0 * self.harvested_crops
            market_pos = [self.GRID_COLS, self.GRID_ROWS // 2]
            self._add_feedback(f"+{coins_earned} Coins!", market_pos, self.COLOR_UI_TEXT, 2.0)
            self.harvested_crops = 0
            # sfx: cash_register.wav

        # 2. Update crop growth
        for plot in self.farm_plots:
            if plot["state"] == self.STATE_PLANTED:
                plot["growth"] += 1
                if plot["growth"] >= self.GROWTH_TIME:
                    plot["state"] = self.STATE_GROWN
                    plot["growth"] = 0
                    # sfx: crop_ready.wav
        
        terminated = self.time_remaining <= 0 or self.score >= self.WIN_SCORE
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _add_feedback(self, text, grid_pos, color, life_multiplier=1.0):
        screen_x = self.GRID_OFFSET_X + grid_pos[0] * self.PLOT_SIZE + self.PLOT_SIZE // 2
        screen_y = self.GRID_OFFSET_Y + grid_pos[1] * self.PLOT_SIZE
        self.action_feedback.append({
            "text": text,
            "pos": [screen_x, screen_y],
            "color": color,
            "life": int(60 * life_multiplier)
        })

    def _draw_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf = font.render(text, True, self.COLOR_UI_SHADOW)
            self.screen.blit(text_surf, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)
        
    def _lerp_color(self, c1, c2, t):
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        pygame.draw.rect(self.screen, self.COLOR_SOIL_BG, (self.GRID_OFFSET_X-10, self.GRID_OFFSET_Y-10, self.GRID_WIDTH+20, self.GRID_HEIGHT+20), border_radius=5)

        # Render all game elements
        # Render Farm Plots
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                plot = self._get_plot(x, y)
                px = self.GRID_OFFSET_X + x * self.PLOT_SIZE
                py = self.GRID_OFFSET_Y + y * self.PLOT_SIZE
                
                pygame.draw.rect(self.screen, self.COLOR_SOIL_PLOT, (px, py, self.PLOT_SIZE, self.PLOT_SIZE))
                pygame.draw.rect(self.screen, self.COLOR_SOIL_BG, (px, py, self.PLOT_SIZE, self.PLOT_SIZE), 1)

                if plot["state"] == self.STATE_PLANTED:
                    progress = plot["growth"] / self.GROWTH_TIME
                    radius = int(2 + progress * (self.PLOT_SIZE / 2 - 6))
                    color = self._lerp_color(self.COLOR_SEED, self.COLOR_PLANT, progress)
                    pygame.gfxdraw.filled_circle(self.screen, px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2, radius, color)
                elif plot["state"] == self.STATE_GROWN:
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    radius = int(self.PLOT_SIZE / 2 - 4 + pulse * 2)
                    color = self._lerp_color(self.COLOR_GROWN, (255, 255, 150), pulse)
                    pygame.gfxdraw.filled_circle(self.screen, px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2, radius, color)
        
        # Render Cursor
        cursor_px = self.GRID_OFFSET_X + self.cursor_pos[0] * self.PLOT_SIZE
        cursor_py = self.GRID_OFFSET_Y + self.cursor_pos[1] * self.PLOT_SIZE
        cursor_surf = pygame.Surface((self.PLOT_SIZE, self.PLOT_SIZE), pygame.SRCALPHA)
        cursor_surf.fill(self.COLOR_CURSOR)
        self.screen.blit(cursor_surf, (cursor_px, cursor_py))
        pygame.draw.rect(self.screen, (255,255,255), (cursor_px, cursor_py, self.PLOT_SIZE, self.PLOT_SIZE), 2)

        # Render Market Stall
        market_x = self.GRID_OFFSET_X + self.GRID_WIDTH + 20
        market_y = self.GRID_OFFSET_Y + self.GRID_HEIGHT // 2 - 40
        pygame.draw.rect(self.screen, self.COLOR_MARKET_RED, (market_x, market_y, 60, 80))
        for i in range(4):
            color = self.COLOR_MARKET_WHITE if i % 2 == 0 else self.COLOR_MARKET_RED
            pygame.draw.rect(self.screen, color, (market_x + i * 15, market_y - 15, 15, 15))
        self._draw_text("SELL", self.font_ui, self.COLOR_UI_TEXT, (market_x + 5, market_y + 85))
        
        # Render UI overlay
        self._draw_text(f"COINS: {self.score}", self.font_ui, self.COLOR_UI_TEXT, (10, 10))
        self._draw_text(f"HELD: {self.harvested_crops}", self.font_ui, self.COLOR_UI_TEXT, (10, 40))
        
        timer_width, timer_height = 200, 20
        timer_x, timer_y = self.WIDTH - timer_width - 10, 10
        time_ratio = self.time_remaining / self.MAX_STEPS
        
        timer_color = self.COLOR_TIMER_GOOD
        if time_ratio < 0.5: timer_color = self.COLOR_TIMER_WARN
        if time_ratio < 0.2: timer_color = self.COLOR_TIMER_BAD
        
        pygame.draw.rect(self.screen, self.COLOR_UI_SHADOW, (timer_x, timer_y, timer_width, timer_height))
        pygame.draw.rect(self.screen, timer_color, (timer_x, timer_y, int(timer_width * time_ratio), timer_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (timer_x, timer_y, timer_width, timer_height), 1)

        # Render Action Feedback
        for fb in self.action_feedback[:]:
            fb['pos'][1] -= 0.5
            fb['life'] -= 1
            alpha = min(255, int(255 * (fb['life'] / 60.0)))
            
            if alpha <= 0:
                self.action_feedback.remove(fb)
                continue
            
            text_surf = self.font_feedback.render(fb['text'], True, fb['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=fb['pos'])
            self.screen.blit(text_surf, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_text = "GOAL REACHED!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            win_color = (100, 255, 100) if self.score >= self.WIN_SCORE else (255, 100, 100)
            
            text_surf = self.font_game_over.render(win_text, True, win_color)
            text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20))
            self.screen.blit(text_surf, text_rect)

            final_score_surf = self.font_ui.render(f"Final Coins: {self.score}", True, self.COLOR_UI_TEXT)
            final_score_rect = final_score_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 40))
            self.screen.blit(final_score_surf, final_score_rect)

        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This block allows you to play the game directly for testing.
    class HumanGameEnv(GameEnv):
        metadata = {"render_modes": ["rgb_array", "human"]}
        def __init__(self, render_mode="rgb_array"):
            super().__init__(render_mode)
            self.render_mode = render_mode
            if self.render_mode == "human":
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                pygame.display.set_caption("Farm Manager")

        def _get_observation(self):
            obs = super()._get_observation()
            if self.render_mode == "human":
                surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                self.human_screen.blit(surf, (0, 0))
                pygame.display.flip()
            return obs

    env = HumanGameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    while not done:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
        
        env.clock.tick(30)

    env.close()