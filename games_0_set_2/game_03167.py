
# Generated: 2025-08-28T07:11:13.006375
# Source Brief: brief_03167.md
# Brief Index: 3167

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to plant the selected crop or harvest a ready one. "
        "Press Shift to cycle through crop types. Move the cursor to the market (bottom right) and press Shift to sell."
    )

    game_description = (
        "A fast-paced farming simulation. Plant crops, wait for them to grow, and harvest them. "
        "Sell your produce at the market to earn gold. Reach 1000 gold before the 60-second timer runs out to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and timing
        self.screen_width = 640
        self.screen_height = 400
        self.fps = 30
        self.max_time_seconds = 60
        self.max_steps = self.max_time_seconds * self.fps

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (20, 30, 25)
        self.COLOR_PLOT_EMPTY = (94, 63, 43)
        self.COLOR_PLOT_OUTLINE = (54, 33, 23)
        self.COLOR_MARKET = (130, 100, 60)
        self.COLOR_MARKET_ROOF = (180, 50, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_GLOW = (0, 200, 255)
        self.COLOR_UI_TEXT = (255, 255, 220)
        self.COLOR_GOLD = (255, 215, 0)
        
        # Game constants
        self.win_gold = 1000
        self.grid_size = (4, 3)
        self.plot_size = 60
        self.plot_padding = 20
        self.grid_width = self.grid_size[0] * (self.plot_size + self.plot_padding) - self.plot_padding
        self.grid_height = self.grid_size[1] * (self.plot_size + self.plot_padding) - self.plot_padding
        self.grid_origin_x = (self.screen_width - self.grid_width) // 2
        self.grid_origin_y = 80

        # Crop definitions: [name, grow_time_steps, value, growing_color, ready_color]
        self.crop_types = [
            {"name": "Carrot", "grow_time": 4 * self.fps, "value": 10, "color_grow": (255, 165, 0), "color_ready": (255, 140, 0)},
            {"name": "Cabbage", "grow_time": 7 * self.fps, "value": 25, "color_grow": (144, 238, 144), "color_ready": (50, 205, 50)},
            {"name": "Pumpkin", "grow_time": 11 * self.fps, "value": 55, "color_grow": (255, 120, 0), "color_ready": (255, 69, 0)},
        ]
        
        self.plot_rects = self._calculate_plot_rects()
        self.market_rect = pygame.Rect(self.screen_width - 120, self.screen_height - 100, 100, 80)
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _calculate_plot_rects(self):
        rects = []
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                px = self.grid_origin_x + x * (self.plot_size + self.plot_padding)
                py = self.grid_origin_y + y * (self.plot_size + self.plot_padding)
                rects.append(pygame.Rect(px, py, self.plot_size, self.plot_size))
        return rects

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.gold = 0
        self.game_over = False
        
        self.plots = [
            {"state": "empty", "growth": 0, "crop_index": -1} 
            for _ in range(self.grid_size[0] * self.grid_size[1])
        ]
        self.inventory = {crop["name"]: 0 for crop in self.crop_types}
        
        self.cursor_pos = [0, 0] # grid x, y
        self.selected_crop_index = 0
        self.previous_action = np.array([0, 0, 0])
        self.move_cooldown = 0
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_actions(action)
        reward += self._update_game_state()
        self._update_particles()
        
        self.steps += 1
        
        terminated = self.steps >= self.max_steps or self.gold >= self.win_gold
        if terminated:
            self.game_over = True
            if self.gold >= self.win_gold:
                reward += 100.0  # Win bonus
            else:
                reward -= 10.0   # Time-out penalty
        
        self.previous_action = action

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_action, shift_action = action
        
        space_pressed = space_action == 1 and self.previous_action[1] == 0
        shift_pressed = shift_action == 1 and self.previous_action[2] == 0

        # --- Movement ---
        if self.move_cooldown <= 0:
            moved = False
            if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1; moved = True
            elif movement == 2 and self.cursor_pos[1] < self.grid_size[1] - 1: self.cursor_pos[1] += 1; moved = True
            elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1; moved = True
            elif movement == 4 and self.cursor_pos[0] < self.grid_size[0] - 1: self.cursor_pos[0] += 1; moved = True
            if moved: self.move_cooldown = self.fps // 6 # Cooldown to make it human-playable
        else:
            self.move_cooldown -=1

        cursor_on_market = self.get_cursor_rect().colliderect(self.market_rect)

        # --- Shift Action (Cycle Crop / Sell) ---
        if shift_pressed:
            if cursor_on_market:
                # Sell
                sell_value = 0
                for crop_def in self.crop_types:
                    name = crop_def["name"]
                    count = self.inventory[name]
                    if count > 0:
                        sell_value += count * crop_def["value"]
                        self.inventory[name] = 0
                if sell_value > 0:
                    self.gold += sell_value
                    # sfx: cash register
                    self._add_particles(self.market_rect.center, self.COLOR_GOLD, 50, life=40, speed=3)
            else:
                # Cycle crop
                self.selected_crop_index = (self.selected_crop_index + 1) % len(self.crop_types)
                # sfx: UI click
                
        # --- Space Action (Plant / Harvest) ---
        if space_pressed and not cursor_on_market:
            plot_index = self.cursor_pos[1] * self.grid_size[0] + self.cursor_pos[0]
            plot = self.plots[plot_index]
            
            if plot["state"] == "empty":
                # Plant
                plot["state"] = "growing"
                plot["crop_index"] = self.selected_crop_index
                # sfx: plant seed
            elif plot["state"] == "ready":
                # Harvest
                crop_name = self.crop_types[plot["crop_index"]]["name"]
                self.inventory[crop_name] += 1
                plot["state"] = "empty"
                plot["growth"] = 0
                plot["crop_index"] = -1
                # sfx: harvest pop
                self._add_particles(self.plot_rects[plot_index].center, (255,255,100), 20, life=20)
                
    def _update_game_state(self):
        reward = 0
        for plot in self.plots:
            if plot["state"] == "growing":
                plot["growth"] += 1
                crop_def = self.crop_types[plot["crop_index"]]
                if plot["growth"] >= crop_def["grow_time"]:
                    plot["state"] = "ready"
                    reward += 0.1 # Small reward for a crop becoming ready
        return reward
        
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    def _add_particles(self, pos, color, count, life=30, speed=2):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5, speed)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'vx': math.cos(angle) * s, 'vy': math.sin(angle) * s,
                'life': life, 'max_life': life, 'color': color
            })
            
    def get_cursor_rect(self):
        plot_index = self.cursor_pos[1] * self.grid_size[0] + self.cursor_pos[0]
        return self.plot_rects[plot_index]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_plots_and_market()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_plots_and_market(self):
        # Draw plots
        for i, rect in enumerate(self.plot_rects):
            pygame.draw.rect(self.screen, self.COLOR_PLOT_OUTLINE, rect.inflate(4, 4), border_radius=8)
            pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, rect, border_radius=8)
            
            plot = self.plots[i]
            if plot["state"] != "empty":
                crop_def = self.crop_types[plot["crop_index"]]
                progress = plot["growth"] / crop_def["grow_time"]
                
                if plot["state"] == "growing":
                    color = crop_def["color_grow"]
                    size = int(self.plot_size * 0.8 * progress)
                    pygame.draw.circle(self.screen, color, rect.center, max(0, size // 2))
                elif plot["state"] == "ready":
                    color = crop_def["color_ready"]
                    size = int(self.plot_size * 0.8)
                    pulse = abs(math.sin(self.steps * 0.2)) * 5
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, max(0, size // 2 + int(pulse)), (*color, 100))
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, max(0, size // 2), color)

        # Draw market
        pygame.draw.rect(self.screen, self.COLOR_MARKET, self.market_rect, border_radius=8)
        pygame.draw.polygon(self.screen, self.COLOR_MARKET_ROOF, [
            (self.market_rect.left - 5, self.market_rect.top),
            (self.market_rect.right + 5, self.market_rect.top),
            (self.market_rect.centerx, self.market_rect.top - 20)
        ])
        market_text = self.font_small.render("SELL", True, self.COLOR_UI_TEXT)
        self.screen.blit(market_text, market_text.get_rect(center=self.market_rect.center))

    def _render_cursor(self):
        rect = self.get_cursor_rect()
        pulse = abs(math.sin(self.steps * 0.15)) * 4
        
        # Glow
        pygame.gfxdraw.rectangle(self.screen, rect.inflate(pulse, pulse), (*self.COLOR_CURSOR_GLOW, 50))
        # Main cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3, border_radius=8)
        
        # Show selected crop near cursor
        crop_def = self.crop_types[self.selected_crop_index]
        crop_color = crop_def["color_ready"]
        pygame.draw.circle(self.screen, crop_color, (rect.right - 10, rect.top + 10), 8)
        pygame.draw.circle(self.screen, (0,0,0), (rect.right - 10, rect.top + 10), 8, 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), 2, color)

    def _render_ui(self):
        # Gold display
        gold_text = self.font_large.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (20, 10))

        # Time bar
        time_ratio = (self.max_steps - self.steps) / self.max_steps
        bar_width = 200
        bar_height = 20
        bar_x = self.screen_width - bar_width - 20
        bar_y = 15
        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        fill_color = (0, 200, 0) if time_ratio > 0.25 else (200, 0, 0)
        pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, bar_width * time_ratio, bar_height), border_radius=5)
        
        # Inventory display
        inv_y = self.screen_height - 30
        for i, crop_def in enumerate(self.crop_types):
            name = crop_def["name"]
            count = self.inventory[name]
            inv_text = self.font_small.render(f"{name}: {count}", True, self.COLOR_UI_TEXT)
            self.screen.blit(inv_text, (20 + i * 120, inv_y))

    def _render_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.gold >= self.win_gold:
            message = "YOU WIN!"
            color = self.COLOR_GOLD
        else:
            message = "TIME'S UP!"
            color = (200, 50, 50)
            
        text = self.font_huge.render(message, True, color)
        text_rect = text.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
        overlay.blit(text, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.gold,
            "steps": self.steps,
            "time_left": self.max_time_seconds - (self.steps / self.fps),
            "inventory": self.inventory,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Farm Frenzy")
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Game loop
    while not terminated:
        # Get keyboard input for manual play
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        # Cap the frame rate
        clock.tick(env.fps)
        
    print(f"Game Over! Final Score: {info['score']}")
    env.close()