
# Generated: 2025-08-27T13:15:40.325014
# Source Brief: brief_00306.md
# Brief Index: 306

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    user_guide = (
        "Controls: Use arrow keys to select a farm plot. Press space to plant a seed on empty soil or harvest a ready crop. "
        "Press shift to sell all harvested crops to the current customer."
    )

    game_description = (
        "Manage a small isometric farm. Plant, harvest, and sell crops to reach 1000 gold before time runs out. "
        "Customers will appear with different price offers."
    )

    auto_advance = False

    # --- Constants ---
    # Game Parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FARM_GRID_SIZE = (4, 4)
    MAX_STEPS = 1200
    WIN_SCORE = 1000
    CROP_GROW_TIME = 60
    CUSTOMER_SPAWN_CHANCE = 0.1
    CUSTOMER_OFFER_MIN, CUSTOMER_OFFER_MAX = 8, 15
    SALE_ANIMATION_DURATION = 30

    # States
    STATE_EMPTY = 0
    STATE_PLANTED = 1
    STATE_READY = 2

    # Colors
    COLOR_BG = (34, 51, 34)
    COLOR_SOIL = (85, 60, 42)
    COLOR_SOIL_DARK = (68, 48, 34)
    COLOR_SELECTOR = (50, 255, 255)
    COLOR_PLANTED = (76, 175, 80)
    COLOR_READY = (255, 235, 59)
    COLOR_UI_TEXT = (255, 255, 240)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_GOLD = (255, 215, 0)
    COLOR_CUSTOMER_BG = (240, 240, 220)
    COLOR_CUSTOMER_TEXT = (50, 50, 40)
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 32)
        self.font_customer = pygame.font.Font(None, 24)
        self.font_sale = pygame.font.Font(None, 28)
        self.font_end_game = pygame.font.Font(None, 64)

        self.render_mode = render_mode
        self.window = None

        # Game State (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_left = None
        self.farm_plots_state = None
        self.farm_plots_timer = None
        self.selected_plot = None
        self.harvested_crops = None
        self.customer_present = None
        self.customer_offer = None
        self.last_sale_info = None
        self.np_random = None

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        
        rows, cols = self.FARM_GRID_SIZE
        self.farm_plots_state = np.full((rows, cols), self.STATE_EMPTY, dtype=np.int8)
        self.farm_plots_timer = np.zeros((rows, cols), dtype=np.int16)
        
        self.selected_plot = (rows // 2, cols // 2)
        self.harvested_crops = 0
        self.customer_present = False
        self.customer_offer = 0
        self.last_sale_info = {"amount": 0, "timer": 0}
        
        self._spawn_customer()

        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Action Handling (Prioritized) ---
        action_taken = False
        # 1. Sell Action
        if shift_pressed and self.customer_present and self.harvested_crops > 0:
            # sfx: cash_register
            gold_earned = self.harvested_crops * self.customer_offer
            self.score += gold_earned
            reward += gold_earned
            self.last_sale_info = {"amount": gold_earned, "timer": self.SALE_ANIMATION_DURATION}
            self.harvested_crops = 0
            self.customer_present = False
            action_taken = True
        # 2. Plant/Harvest Action
        elif space_pressed:
            r, c = self.selected_plot
            if self.farm_plots_state[r, c] == self.STATE_EMPTY:
                # sfx: plant_seed
                self.farm_plots_state[r, c] = self.STATE_PLANTED
                self.farm_plots_timer[r, c] = self.CROP_GROW_TIME
                action_taken = True
            elif self.farm_plots_state[r, c] == self.STATE_READY:
                # sfx: harvest
                self.farm_plots_state[r, c] = self.STATE_EMPTY
                self.harvested_crops += 1
                action_taken = True
        # 3. Movement Action
        elif movement != 0:
            # sfx: cursor_move
            r, c = self.selected_plot
            rows, cols = self.FARM_GRID_SIZE
            if movement == 1: r = (r - 1) % rows  # Up
            elif movement == 2: r = (r + 1) % rows  # Down
            elif movement == 3: c = (c - 1) % cols  # Left
            elif movement == 4: c = (c + 1) % cols  # Right
            self.selected_plot = (r, c)
            action_taken = True

        # --- Game State Update ---
        self.steps += 1
        self.time_left -= 1
        if self.last_sale_info["timer"] > 0:
            self.last_sale_info["timer"] -= 1

        # Update crop growth
        growing_plots = self.farm_plots_state == self.STATE_PLANTED
        self.farm_plots_timer[growing_plots] -= 1
        ready_plots = (self.farm_plots_timer <= 0) & growing_plots
        self.farm_plots_state[ready_plots] = self.STATE_READY
        if np.any(ready_plots):
            pass # sfx: crop_ready_ding

        # Update customer
        if not self.customer_present:
            if self.np_random.random() < self.CUSTOMER_SPAWN_CHANCE:
                self._spawn_customer()

        # --- Termination Check ---
        terminated = (self.score >= self.WIN_SCORE) or (self.time_left <= 0)
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100.0 # Victory bonus
            else:
                reward -= 100.0 # Loss penalty
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_customer(self):
        self.customer_present = True
        self.customer_offer = self.np_random.integers(self.CUSTOMER_OFFER_MIN, self.CUSTOMER_OFFER_MAX + 1)
        # sfx: customer_arrives

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "harvested_crops": self.harvested_crops,
            "customer_offer": self.customer_offer if self.customer_present else 0,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        # human mode is handled in step/reset
        return None

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Isometric Farmer")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        canvas.fill(self.COLOR_BG)
        self._render_game(canvas)
        self._render_ui(canvas)
        if self.game_over:
            self._render_end_screen(canvas)

        if self.window is not None:
            self.window.blit(canvas, (0, 0))
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        pygame.quit()

    # --- Rendering Helpers ---

    def _iso_to_screen(self, r, c, tile_w=64, tile_h=32, offset_x=0, offset_y=0):
        screen_x = (self.SCREEN_WIDTH / 2) + (c - r) * (tile_w / 2) + offset_x
        screen_y = 120 + (c + r) * (tile_h / 2) + offset_y
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, color, r, c, tile_w=64, tile_h=32):
        x, y = self._iso_to_screen(r, c, tile_w, tile_h)
        points = [
            (x, y - tile_h // 2),
            (x + tile_w // 2, y),
            (x, y + tile_h // 2),
            (x - tile_w // 2, y),
        ]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _draw_iso_rect_top(self, surface, color, r, c, height, tile_w=64, tile_h=32):
        x, y = self._iso_to_screen(r, c, tile_w, tile_h, offset_y=-height)
        points = [
            (x, y - tile_h // 2),
            (x + tile_w // 2, y),
            (x, y + tile_h // 2),
            (x - tile_w // 2, y),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)

    def _render_game(self, surface=None):
        if surface is None:
            surface = self.screen

        # Draw farm plots
        rows, cols = self.FARM_GRID_SIZE
        for r in range(rows):
            for c in range(cols):
                # Draw base soil
                self._draw_iso_poly(surface, self.COLOR_SOIL_DARK, r, c)
                self._draw_iso_rect_top(surface, self.COLOR_SOIL, r, c, height=5)

                # Draw crops
                state = self.farm_plots_state[r, c]
                if state == self.STATE_PLANTED:
                    growth_ratio = 1.0 - (self.farm_plots_timer[r, c] / self.CROP_GROW_TIME)
                    radius = int(3 + 10 * growth_ratio)
                    x, y = self._iso_to_screen(r, c, offset_y=-10)
                    pygame.gfxdraw.aacircle(surface, x, y, radius, self.COLOR_PLANTED)
                    pygame.gfxdraw.filled_circle(surface, x, y, radius, self.COLOR_PLANTED)
                elif state == self.STATE_READY:
                    x, y = self._iso_to_screen(r, c, offset_y=-15)
                    pygame.gfxdraw.aacircle(surface, x, y, 14, self.COLOR_READY)
                    pygame.gfxdraw.filled_circle(surface, x, y, 14, self.COLOR_READY)

        # Draw selector
        sel_r, sel_c = self.selected_plot
        x, y = self._iso_to_screen(sel_r, sel_c, tile_w=68, tile_h=34, offset_y=-5)
        points = [
            (x, y - 34 // 2), (x + 68 // 2, y),
            (x, y + 34 // 2), (x - 68 // 2, y)
        ]
        pygame.draw.aalines(surface, self.COLOR_SELECTOR, True, points, 2)
        
    def _render_ui(self, surface=None):
        if surface is None:
            surface = self.screen

        # --- Top Bar ---
        bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 50)
        s = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        surface.blit(s, (0, 0))
        pygame.draw.line(surface, self.COLOR_UI_TEXT, (0, 50), (self.SCREEN_WIDTH, 50), 1)

        # Gold/Score
        score_text = self.font_ui.render(f"Gold: {self.score}", True, self.COLOR_GOLD)
        surface.blit(score_text, (15, 12))

        # Time
        time_text = self.font_ui.render(f"Time: {self.time_left}", True, self.COLOR_UI_TEXT)
        surface.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 15, 12))

        # --- Bottom Bar (Inventory/Customer) ---
        bar_rect_bottom = pygame.Rect(0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50)
        s_bottom = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        s_bottom.fill(self.COLOR_UI_BG)
        surface.blit(s_bottom, (0, self.SCREEN_HEIGHT - 50))
        pygame.draw.line(surface, self.COLOR_UI_TEXT, (0, self.SCREEN_HEIGHT-50), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT-50), 1)

        # Harvested Crops
        crop_text = self.font_ui.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_READY)
        surface.blit(crop_text, (15, self.SCREEN_HEIGHT - 38))

        # Customer Info
        if self.customer_present:
            customer_str = f"Customer Offer: {self.customer_offer} G/crop"
            customer_color = self.COLOR_GOLD
        else:
            customer_str = "Waiting for customer..."
            customer_color = self.COLOR_UI_TEXT
        
        customer_text = self.font_ui.render(customer_str, True, customer_color)
        surface.blit(customer_text, (self.SCREEN_WIDTH - customer_text.get_width() - 15, self.SCREEN_HEIGHT - 38))

        # Sale Animation
        if self.last_sale_info["timer"] > 0:
            alpha = int(255 * (self.last_sale_info["timer"] / self.SALE_ANIMATION_DURATION))
            y_offset = int(40 * (1 - (self.last_sale_info["timer"] / self.SALE_ANIMATION_DURATION)))
            sale_str = f"+{self.last_sale_info['amount']} G"
            sale_surf = self.font_sale.render(sale_str, True, self.COLOR_GOLD)
            sale_surf.set_alpha(alpha)
            pos_x = self.SCREEN_WIDTH - customer_text.get_width() - 15
            pos_y = self.SCREEN_HEIGHT - 75 - y_offset
            surface.blit(sale_surf, (pos_x, pos_y))

    def _render_end_screen(self, surface=None):
        if surface is None:
            surface = self.screen
        
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.score >= self.WIN_SCORE:
            msg = "YOU WIN!"
            color = self.COLOR_GOLD
        else:
            msg = "TIME'S UP!"
            color = (200, 50, 50)
            
        text_surf = self.font_end_game.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        overlay.blit(text_surf, text_rect)
        surface.blit(overlay, (0, 0))

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    terminated = False
    
    print("\n" + "="*30)
    print("Isometric Farmer - Manual Test")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # Default action is no-op
        action = np.array([0, 0, 0]) 
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    action[2] = 1
        
        if terminated:
            break

        # Only step if an action was taken
        if np.any(action):
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step: {info['steps']}, Gold: {info['score']}, Reward: {reward:.2f}, Terminated: {terminated}")

    print("Game Over!")
    pygame.time.wait(2000)
    env.close()