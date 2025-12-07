
# Generated: 2025-08-27T13:33:33.222555
# Source Brief: brief_00404.md
# Brief Index: 404

        
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
        "Controls: Use arrows to move the cursor. Press space to plant and shift to harvest. "
        "Move to the market stall and press shift to sell all harvested crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a farm to earn 1000 coins before time runs out. Plant crops, "
        "wait for them to grow, harvest them, and sell them at the market."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 12, 6
    PLOT_SIZE = 40
    GRID_WIDTH = GRID_COLS * PLOT_SIZE
    GRID_HEIGHT = GRID_ROWS * PLOT_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 30

    MAX_STEPS = 6000
    WIN_COINS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 25)
    COLOR_GRID_BG = (87, 56, 34)
    COLOR_PLOT_EMPTY = (139, 69, 19)
    COLOR_MARKET = (70, 130, 180)
    COLOR_MARKET_ROOF = (210, 105, 30)
    COLOR_CURSOR = (255, 215, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_SHADOW = (0, 0, 0)
    COLOR_OVERLAY = (0, 0, 0, 180)

    # --- Crop Definitions ---
    CROP_TYPES = {
        "carrot": {"growth_time": 200, "value": 2, "color_ripe": (255, 140, 0), "color_growing": (0, 150, 0)},
        "cabbage": {"growth_time": 350, "value": 5, "color_ripe": (124, 252, 0), "color_growing": (34, 139, 34)},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_msg = pygame.font.SysFont("sans", 48, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        
        self.particles = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.coins = 0
        self.game_over = False
        self.win = False
        self.time_remaining = self.MAX_STEPS

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.market_pos = [self.GRID_COLS - 1, self.GRID_ROWS // 2]

        self.farm_grid = [
            [self._create_empty_plot() for _ in range(self.GRID_COLS)]
            for _ in range(self.GRID_ROWS)
        ]

        self.harvested_crops = {crop_name: 0 for crop_name in self.CROP_TYPES}
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01  # Small penalty for passing time

        # 1. Update cursor position
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        cx, cy = self.cursor_pos

        # 2. Handle actions
        if cx == self.market_pos[0] and cy == self.market_pos[1]:
            if shift_held:
                sell_reward, sold_something = self._sell_crops()
                if sold_something:
                    reward += sell_reward
                    # Sound: cha-ching!
        else:
            plot = self.farm_grid[cy][cx]
            if space_held and plot["state"] == "empty":
                self._plant_crop(cx, cy)
                # Sound: pop!
            elif shift_held and plot["state"] == "planted":
                if self._harvest_crop(cx, cy):
                    reward += 0.1
                    # Sound: pluck!

        # 3. Update game state (time passes)
        self.steps += 1
        self.time_remaining -= 1
        self._update_crops()
        self._update_particles()

        # 4. Check termination conditions
        terminated = False
        if self.coins >= self.WIN_COINS:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
        elif self.time_remaining <= 0:
            self.game_over = True
            self.win = False
            terminated = True
            reward -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.coins, "steps": self.steps}

    # --- Helper Methods: Game Logic ---
    def _create_empty_plot(self):
        return {"state": "empty", "type": None, "growth": 0}

    def _plant_crop(self, x, y):
        crop_name = "carrot" if self.rng.random() < 0.5 else "cabbage"
        self.farm_grid[y][x] = {"state": "planted", "type": crop_name, "growth": 0}
        px, py = self._grid_to_pixel(x, y)
        self._create_particles(px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2, 10, self.COLOR_PLOT_EMPTY)

    def _harvest_crop(self, x, y):
        plot = self.farm_grid[y][x]
        crop_info = self.CROP_TYPES[plot["type"]]
        if plot["growth"] >= crop_info["growth_time"]:
            self.harvested_crops[plot["type"]] += 1
            self.farm_grid[y][x] = self._create_empty_plot()
            px, py = self._grid_to_pixel(x, y)
            self._create_particles(px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2, 15, crop_info["color_ripe"])
            return True
        return False

    def _sell_crops(self):
        earnings = 0
        total_sold = 0
        for crop_name, count in self.harvested_crops.items():
            if count > 0:
                earnings += count * self.CROP_TYPES[crop_name]["value"]
                total_sold += count
        
        if earnings > 0:
            self.coins += earnings
            self.harvested_crops = {crop_name: 0 for crop_name in self.CROP_TYPES}
            px, py = self._grid_to_pixel(self.market_pos[0], self.market_pos[1])
            self._create_particles(px + self.PLOT_SIZE // 2, py, 30, self.COLOR_CURSOR, is_coin=True)
            return 1.0 * total_sold, True
        return 0, False

    def _update_crops(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot = self.farm_grid[r][c]
                if plot["state"] == "planted":
                    crop_info = self.CROP_TYPES[plot["type"]]
                    if plot["growth"] < crop_info["growth_time"]:
                        plot["growth"] += 1
    
    # --- Helper Methods: Rendering ---
    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, 
                         (self.GRID_X_OFFSET - 5, self.GRID_Y_OFFSET - 5, 
                          self.GRID_WIDTH + 10, self.GRID_HEIGHT + 10), border_radius=5)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                px, py = self._grid_to_pixel(c, r)
                
                if c == self.market_pos[0] and r == self.market_pos[1]:
                    self._render_market(px, py)
                else:
                    self._render_plot(self.farm_grid[r][c], px, py)
                
                pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (px, py, self.PLOT_SIZE, self.PLOT_SIZE), 1)

        self._render_cursor()
        self._render_particles()

    def _render_market(self, px, py):
        pygame.draw.rect(self.screen, self.COLOR_MARKET, (px, py, self.PLOT_SIZE, self.PLOT_SIZE))
        pygame.draw.rect(self.screen, self.COLOR_MARKET_ROOF, (px, py, self.PLOT_SIZE, 10))
        for i in range(0, self.PLOT_SIZE, 8):
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, (px + i, py), (px + i + 4, py + 10), 2)
        
        carrot_text = f"C:{self.harvested_crops['carrot']}"
        cabbage_text = f"A:{self.harvested_crops['cabbage']}"
        self._draw_text(carrot_text, (px + 3, py + 12), self.font_small, self.CROP_TYPES['carrot']['color_ripe'])
        self._draw_text(cabbage_text, (px + 3, py + 26), self.font_small, self.CROP_TYPES['cabbage']['color_ripe'])

    def _render_plot(self, plot, px, py):
        pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, (px, py, self.PLOT_SIZE, self.PLOT_SIZE))
        if plot["state"] == "planted":
            self._render_crop(plot, px, py)
            
    def _render_crop(self, plot, px, py):
        crop_info = self.CROP_TYPES[plot["type"]]
        growth_ratio = plot["growth"] / crop_info["growth_time"]
        center_x, center_y = px + self.PLOT_SIZE // 2, py + self.PLOT_SIZE // 2

        if growth_ratio >= 1.0:
            color = crop_info["color_ripe"]
            size = int(self.PLOT_SIZE * 0.4)
            pygame.draw.circle(self.screen, color, (center_x, center_y), size)
            shine_pos = (center_x + size // 2, center_y - size // 2)
            pygame.gfxdraw.filled_circle(self.screen, shine_pos[0], shine_pos[1], 3, (255, 255, 255, 150))
        else:
            seedling_size = 2
            max_size = int(self.PLOT_SIZE * 0.35)
            size = int(seedling_size + (max_size - seedling_size) * growth_ratio)
            color = crop_info["color_growing"]
            pygame.draw.circle(self.screen, color, (center_x, center_y), size)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        px, py = self._grid_to_pixel(cx, cy)
        
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        alpha = 100 + 155 * pulse
        
        cursor_surf = pygame.Surface((self.PLOT_SIZE, self.PLOT_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, (*self.COLOR_CURSOR, int(alpha)), (0, 0, self.PLOT_SIZE, self.PLOT_SIZE), 4, border_radius=4)
        self.screen.blit(cursor_surf, (px, py))

    def _render_ui(self):
        coin_text = f"COINS: {self.coins} / {self.WIN_COINS}"
        self._draw_text(coin_text, (10, 10), self.font_ui, self.COLOR_UI_TEXT)

        time_text = f"TIME: {self.time_remaining}"
        text_width = self.font_ui.size(time_text)[0]
        self._draw_text(time_text, (self.SCREEN_WIDTH - text_width - 10, 10), self.font_ui, self.COLOR_UI_TEXT)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "TIME'S UP!"
            msg_width, msg_height = self.font_msg.size(msg)
            self._draw_text(msg, (self.SCREEN_WIDTH // 2 - msg_width // 2, self.SCREEN_HEIGHT // 2 - msg_height // 2), self.font_msg, self.COLOR_UI_TEXT)

    def _draw_text(self, text, pos, font, color):
        shadow_img = font.render(text, True, self.COLOR_UI_SHADOW)
        text_img = font.render(text, True, color)
        self.screen.blit(shadow_img, (pos[0] + 2, pos[1] + 2))
        self.screen.blit(text_img, pos)

    def _grid_to_pixel(self, grid_x, grid_y):
        return self.GRID_X_OFFSET + grid_x * self.PLOT_SIZE, self.GRID_Y_OFFSET + grid_y * self.PLOT_SIZE

    # --- Helper Methods: Particles ---
    def _create_particles(self, x, y, count, color, is_coin=False):
        for _ in range(count):
            if is_coin:
                angle = self.rng.random() * math.pi - math.pi # Upward arc
                speed = self.rng.random() * 3 + 2
            else:
                angle = self.rng.random() * 2 * math.pi
                speed = self.rng.random() * 2 + 1
            
            lifespan = self.rng.integers(15, 30)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "lifespan": lifespan, "max_lifespan": lifespan, "color": color, "is_coin": is_coin
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            if p["is_coin"]:
                p["vy"] += 0.2 # Gravity for coins
            p["lifespan"] -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
            alpha = max(0, alpha)
            size = 3 if p["is_coin"] else 2
            pygame.gfxdraw.filled_circle(self.screen, int(p["x"]), int(p["y"]), size, (*p["color"], alpha))

    # --- Implementation Validation ---
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