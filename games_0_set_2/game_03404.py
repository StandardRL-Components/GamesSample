
# Generated: 2025-08-27T23:15:37.792104
# Source Brief: brief_03404.md
# Brief Index: 3404

        
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
        "Controls: Use arrow keys to select a plot. Press Space to plant or harvest. Hold Shift to sell produce."
    )

    game_description = (
        "Manage an isometric farm. Plant seeds, harvest crops, and sell them at the barn to earn 1000 coins before time runs out."
    )

    auto_advance = False
    
    # --- Constants ---
    # Game parameters
    GRID_WIDTH = 5
    GRID_HEIGHT = 5
    MAX_TIME = 3000
    WIN_COINS = 1000
    CROP_GROW_TIME = 20
    STARTING_SEEDS = 10
    SELL_PRICE = 10
    SEEDS_PER_SALE = 2

    # Plot states
    STATE_EMPTY = 0
    STATE_GROWING = 1
    STATE_READY = 2
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_SOIL = (87, 56, 42)
    COLOR_SOIL_DARK = (67, 40, 28)
    COLOR_GRASS = (60, 120, 60)
    COLOR_BARN_RED = (180, 40, 40)
    COLOR_BARN_ROOF = (90, 90, 90)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_SEEDLING = (100, 200, 100)
    COLOR_GROWN = (255, 220, 80)
    COLOR_TEXT = (240, 240, 240)
    COLOR_UI_BG = (30, 50, 70, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_small = self._create_pixel_font(1)
        self.font_large = self._create_pixel_font(2)
        
        self.iso_tile_w = 32
        self.iso_tile_h = 16
        self.grid_origin_x = self.width // 2
        self.grid_origin_y = 140
        
        self.farm_plots = []
        self.selected_plot_idx = 0
        self.steps = 0
        self.coins = 0
        self.seeds = 0
        self.harvested_produce = 0
        self.time_remaining = 0
        self.game_over = False
        self.last_action_reward = 0
        self.particles = []
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.coins = 0
        self.seeds = self.STARTING_SEEDS
        self.harvested_produce = 0
        self.time_remaining = self.MAX_TIME
        self.game_over = False
        self.last_action_reward = 0
        self.particles = []
        
        self.farm_plots = [{'state': self.STATE_EMPTY, 'growth': 0} for _ in range(self.GRID_WIDTH * self.GRID_HEIGHT)]
        self.selected_plot_idx = self.GRID_WIDTH * self.GRID_HEIGHT // 2
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        self.time_remaining -= 1
        reward = 0
        action_taken = False

        # 1. Handle player actions
        if movement != 0:
            row, col = self.selected_plot_idx // self.GRID_WIDTH, self.selected_plot_idx % self.GRID_WIDTH
            if movement == 1 and row > 0: self.selected_plot_idx -= self.GRID_WIDTH
            elif movement == 2 and row < self.GRID_HEIGHT - 1: self.selected_plot_idx += self.GRID_WIDTH
            elif movement == 3 and col > 0: self.selected_plot_idx -= 1
            elif movement == 4 and col < self.GRID_WIDTH - 1: self.selected_plot_idx += 1
            action_taken = True

        plot = self.farm_plots[self.selected_plot_idx]
        plot_pos_screen = self._iso_to_screen(self.selected_plot_idx % self.GRID_WIDTH, self.selected_plot_idx // self.GRID_WIDTH)

        if space_pressed:
            if plot['state'] == self.STATE_EMPTY and self.seeds > 0:
                plot['state'] = self.STATE_GROWING
                plot['growth'] = 0
                self.seeds -= 1
                action_taken = True
                # sfx: plant_seed
                self._add_particles(plot_pos_screen, 5, self.COLOR_SOIL, 1.5)
            elif plot['state'] == self.STATE_READY:
                plot['state'] = self.STATE_EMPTY
                plot['growth'] = 0
                self.harvested_produce += 1
                reward += 1.0
                action_taken = True
                # sfx: harvest_pop
                self._add_particles(plot_pos_screen, 10, self.COLOR_GROWN, 2.5)

        if shift_pressed and self.harvested_produce > 0:
            self.coins += self.harvested_produce * self.SELL_PRICE
            self.seeds += (self.harvested_produce // self.SEEDS_PER_SALE)
            reward += 10.0
            action_taken = True
            # sfx: cha-ching
            barn_pos = (self.grid_origin_x - self.iso_tile_w * (self.GRID_WIDTH / 2 + 1.5), self.grid_origin_y)
            self._add_particles(barn_pos, 20, (255, 215, 0), 4)
            self.harvested_produce = 0

        # 2. Update game state (crop growth)
        for p in self.farm_plots:
            if p['state'] == self.STATE_GROWING:
                p['growth'] += 1
                if p['growth'] >= self.CROP_GROW_TIME:
                    p['state'] = self.STATE_READY
        
        # 3. Calculate rewards and check termination
        if not action_taken:
            reward -= 0.1
        
        self.last_action_reward = reward
        terminated = self._check_termination()
        
        if terminated:
            if self.coins >= self.WIN_COINS:
                reward += 100.0
            else:
                reward -= 100.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.coins >= self.WIN_COINS or self.time_remaining <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_GRASS)
        self._render_game()
        self._update_and_render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "coins": self.coins,
            "seeds": self.seeds,
            "harvested": self.harvested_produce,
            "time_remaining": self.time_remaining,
            "steps": self.steps,
        }
        
    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.grid_origin_x + (grid_x - grid_y) * self.iso_tile_w
        screen_y = self.grid_origin_y + (grid_x + grid_y) * self.iso_tile_h
        return int(screen_x), int(screen_y)

    def _render_game(self):
        # Draw barn
        barn_base_x = self.grid_origin_x - self.iso_tile_w * (self.GRID_WIDTH / 2 + 2)
        barn_base_y = self.grid_origin_y - self.iso_tile_h * 2
        pygame.draw.rect(self.screen, self.COLOR_BARN_RED, (barn_base_x - 30, barn_base_y, 60, 50))
        pygame.draw.polygon(self.screen, self.COLOR_BARN_ROOF, [
            (barn_base_x - 35, barn_base_y),
            (barn_base_x + 35, barn_base_y),
            (barn_base_x, barn_base_y - 20)
        ])

        # Draw plots from back to front
        for i in range(self.GRID_WIDTH * self.GRID_HEIGHT):
            grid_y, grid_x = i // self.GRID_WIDTH, i % self.GRID_WIDTH
            plot_center_x, plot_center_y = self._iso_to_screen(grid_x, grid_y)
            
            plot_data = self.farm_plots[i]
            self._draw_iso_plot(self.screen, plot_center_x, plot_center_y, plot_data)

        # Draw cursor
        cursor_y, cursor_x = self.selected_plot_idx // self.GRID_WIDTH, self.selected_plot_idx % self.GRID_WIDTH
        cursor_center_x, cursor_center_y = self._iso_to_screen(cursor_x, cursor_y)
        self._draw_iso_rect(self.screen, cursor_center_x, cursor_center_y, self.COLOR_CURSOR, is_cursor=True)

    def _draw_iso_plot(self, surface, x, y, plot_data):
        self._draw_iso_rect(surface, x, y, self.COLOR_SOIL, is_cursor=False)
        
        state = plot_data['state']
        if state == self.STATE_GROWING or state == self.STATE_READY:
            progress = min(1.0, plot_data['growth'] / self.CROP_GROW_TIME)
            
            if state == self.STATE_GROWING:
                color = self.COLOR_SEEDLING
                size = 2 + int(progress * 8)
                y_offset = -size // 2 - 2
            else: # READY
                color = self.COLOR_GROWN
                size = 10
                y_offset = -size // 2 - 4
            
            # Simple stalk
            pygame.draw.line(surface, (30, 100, 30), (x, y - 2), (x, y + y_offset), 2)
            # Crop head
            pygame.gfxdraw.filled_circle(surface, x, y + y_offset, size, color)
            pygame.gfxdraw.aacircle(surface, x, y + y_offset, size, color)

    def _draw_iso_rect(self, surface, x, y, color, is_cursor):
        w, h = self.iso_tile_w, self.iso_tile_h
        points = [
            (x, y - h), (x + w, y), (x, y + h), (x - w, y)
        ]
        if is_cursor:
            s = pygame.Surface((w*2, h*2), pygame.SRCALPHA)
            pygame.draw.polygon(s, color, [
                (p[0]-x+w, p[1]-y+h) for p in points
            ])
            surface.blit(s, (x-w, y-h))
        else:
            pygame.gfxdraw.filled_polygon(surface, points, color)
            # Add a darker top-left edge for 3D effect
            pygame.draw.aaline(surface, self.COLOR_SOIL_DARK, points[0], points[3])
            pygame.draw.aaline(surface, self.COLOR_SOIL_DARK, points[3], points[2])

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            p['radius'] -= 0.05
            
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)
            else:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                radius = int(p['radius'])
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p['color'])

    def _add_particles(self, pos, count, color, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5 * speed, speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * s, math.sin(angle) * s - 1],
                'lifespan': random.randint(20, 40),
                'radius': random.uniform(2, 5),
                'color': color
            })

    def _render_ui(self):
        ui_elements = [
            (f"COINS: {self.coins}", (10, 10)),
            (f"SEEDS: {self.seeds}", (10, 30)),
            (f"PRODUCE: {self.harvested_produce}", (10, 50)),
            (f"TIME: {self.time_remaining}", (self.width - 120, 10)),
        ]
        
        for text, pos in ui_elements:
            self._render_pixel_text(text, self.font_large, pos, self.COLOR_TEXT)

        if self.game_over:
            s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            
            msg = "YOU WIN!" if self.coins >= self.WIN_COINS else "TIME'S UP!"
            text_surf, text_rect = self._render_pixel_text(msg, self._create_pixel_font(4), (0,0), self.COLOR_GROWN, return_surface=True)
            text_rect.center = (self.width // 2, self.height // 2 - 20)
            self.screen.blit(text_surf, text_rect)
            
            final_score_text = f"FINAL COINS: {self.coins}"
            text_surf, text_rect = self._render_pixel_text(final_score_text, self.font_large, (0,0), self.COLOR_TEXT, return_surface=True)
            text_rect.center = (self.width // 2, self.height // 2 + 20)
            self.screen.blit(text_surf, text_rect)

    def _create_pixel_font(self, scale):
        font_map = {
            'A': [' 0 ', '0 0', '000', '0 0', '0 0'], 'B': ['00 ', '0 0', '00 ', '0 0', '00 '],
            'C': [' 00', '0  ', '0  ', '0  ', ' 00'], 'D': ['00 ', '0 0', '0 0', '0 0', '00 '],
            'E': ['000', '0  ', '00 ', '0  ', '000'], 'F': ['000', '0  ', '00 ', '0  ', '0  '],
            'G': [' 00', '0  ', '0 0', '0 0', ' 00'], 'H': ['0 0', '0 0', '000', '0 0', '0 0'],
            'I': ['000', ' 0 ', ' 0 ', ' 0 ', '000'], 'J': ['  0', '  0', '  0', '0 0', ' 0 '],
            'K': ['0 0', '0 0', '00 ', '0 0', '0 0'], 'L': ['0  ', '0  ', '0  ', '0  ', '000'],
            'M': ['0 0', '000', '0 0', '0 0', '0 0'], 'N': ['0 0', '00 0', '0 00', '0 0', '0 0'],
            'O': [' 0 ', '0 0', '0 0', '0 0', ' 0 '], 'P': ['00 ', '0 0', '00 ', '0  ', '0  '],
            'Q': [' 0 ', '0 0', '0 0', '0 0', ' 00'], 'R': ['00 ', '0 0', '00 ', '0 0', '0 0'],
            'S': [' 00', '0  ', ' 0 ', '  0', '00 '], 'T': ['000', ' 0 ', ' 0 ', ' 0 ', ' 0 '],
            'U': ['0 0', '0 0', '0 0', '0 0', ' 0 '], 'V': ['0 0', '0 0', '0 0', ' 0 ', ' 0 '],
            'W': ['0 0', '0 0', '0 0', '000', '0 0'], 'X': ['0 0', ' 0 ', ' 0 ', ' 0 ', '0 0'],
            'Y': ['0 0', '0 0', ' 0 ', ' 0 ', ' 0 '], 'Z': ['000', '  0', ' 0 ', '0  ', '000'],
            '0': [' 0 ', '0 0', '0 0', '0 0', ' 0 '], '1': [' 0 ', '00 ', ' 0 ', ' 0 ', '000'],
            '2': ['00 ', '  0', ' 0 ', '0  ', '000'], '3': ['00 ', '  0', '00 ', '  0', '00 '],
            '4': ['0 0', '0 0', '000', '  0', '  0'], '5': ['000', '0  ', '00 ', '  0', '00 '],
            '6': [' 0 ', '0  ', '00 ', '0 0', ' 0 '], '7': ['000', '  0', '  0', ' 0 ', ' 0 '],
            '8': [' 0 ', '0 0', ' 0 ', '0 0', ' 0 '], '9': [' 0 ', '0 0', ' 00', '  0', ' 0 '],
            ':': [' ', '0', ' ', '0', ' '], ' ': ['   ', '   ', '   ', '   ', '   '],
            '!': ['0', '0', '0', ' ', '0'],
        }
        return {'map': font_map, 'scale': scale, 'spacing': 1}

    def _render_pixel_text(self, text, font, pos, color, return_surface=False):
        font_map, scale, spacing = font['map'], font['scale'], font['spacing']
        x, y = pos
        char_width = 3 * scale + spacing * scale
        
        if return_surface:
            width = sum(char_width for char in text.upper())
            height = 5 * scale
            surf = pygame.Surface((width, height), pygame.SRCALPHA)
            current_x = 0
            for char in text.upper():
                if char in font_map:
                    for row_idx, row in enumerate(font_map[char]):
                        for col_idx, pixel in enumerate(row):
                            if pixel == '0':
                                pygame.draw.rect(surf, color, (current_x + col_idx * scale, row_idx * scale, scale, scale))
                current_x += char_width
            return surf, surf.get_rect()
        else:
            for char in text.upper():
                if char in font_map:
                    for row_idx, row in enumerate(font_map[char]):
                        for col_idx, pixel in enumerate(row):
                            if pixel == '0':
                                pygame.draw.rect(self.screen, color, (x + col_idx * scale, y + row_idx * scale, scale, scale))
                x += char_width

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Override the screen for display
    env.screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Farm Manager")

    print(env.user_guide)
    
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Only step if an action is taken
        if any(a != 0 for a in action):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")
            if terminated:
                print("Game Over!")
        
        # Always render the current state
        pygame.surfarray.blit_array(env.screen, np.transpose(obs, (1, 0, 2)))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- GAME RESET ---")

        env.clock.tick(30) # Limit FPS for human playability
        
    pygame.quit()