
# Generated: 2025-08-28T05:23:38.176862
# Source Brief: brief_05560.md
# Brief Index: 5560

        
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
        "Controls: Arrow keys to move cursor. Space to plant/harvest. "
        "Shift to cycle selected crop. Move to the market stall to sell (Space)."
    )

    game_description = (
        "Manage an isometric farm. Plant crops, wait for them to grow, then harvest them. "
        "Sell your harvest at the market stall to earn money. Reach $1000 to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.FONT_S = pygame.font.Font(None, 24)
        self.FONT_M = pygame.font.Font(None, 32)
        self.FONT_L = pygame.font.Font(None, 64)

        # --- Game Constants ---
        self.GRID_SIZE = (10, 10)
        self.SCREEN_SIZE = (640, 400)
        self.WIN_SCORE = 1000
        self.STARTING_MONEY = 100
        self.MAX_STEPS = 300 * 30  # 300 seconds at 30fps

        self.MARKET_POS = (9, 9)

        self.CROP_DATA = [
            {'name': 'Wheat', 'cost': 5, 'growth_time': 60 * 2, 'sell_price': 10, 'color': (240, 220, 50), 'leaf_color': (255, 240, 100)},
            {'name': 'Corn', 'cost': 10, 'growth_time': 80 * 2, 'sell_price': 15, 'color': (50, 180, 50), 'leaf_color': (80, 220, 80)},
            {'name': 'Carrots', 'cost': 15, 'growth_time': 100 * 2, 'sell_price': 20, 'color': (255, 140, 0), 'leaf_color': (0, 200, 80)},
        ]

        # --- Visuals ---
        self.TILE_W, self.TILE_H = 52, 26
        self.ORIGIN_X, self.ORIGIN_Y = self.SCREEN_SIZE[0] // 2, 110
        
        self.COLOR_BG = (54, 138, 75)
        self.COLOR_GRID = (85, 159, 101)
        self.COLOR_DIRT = (139, 105, 75)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_MARKET = (180, 80, 80)
        self.COLOR_TEXT = (255, 255, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_UI_BG = (0, 0, 0, 128)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.money = 0
        self.game_over = False
        self.win_state = False
        self.cursor_pos = [0, 0]
        self.plots = []
        self.inventory = {}
        self.selected_plant_idx = 0
        self.last_action = [0, 0, 0]
        self.move_cooldown = 0
        self.particles = []
        self.floating_texts = []
        self.cumulative_reward = 0.0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.money = self.STARTING_MONEY
        self.game_over = False
        self.win_state = False
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.plots = [[{'type': None, 'growth': 0} for _ in range(self.GRID_SIZE[0])] for _ in range(self.GRID_SIZE[1])]
        self.inventory = {crop['name']: 0 for crop in self.CROP_DATA}
        self.selected_plant_idx = 0
        self.last_action = [0, 0, 0]
        self.move_cooldown = 0
        self.particles = []
        self.floating_texts = []
        self.cumulative_reward = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for passing time
        self.steps += 1
        
        # --- Update Game State (Time-based) ---
        self._update_plots()
        self._update_particles()
        self._update_floating_texts()

        # --- Handle Input ---
        reward += self._handle_input(action)
        self.cumulative_reward += reward

        # --- Check Termination ---
        terminated = False
        if self.money >= self.WIN_SCORE:
            reward += 100
            self.cumulative_reward += 100
            terminated = True
            self.game_over = True
            self.win_state = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 10
            self.cumulative_reward -= 10
            terminated = True
            self.game_over = True
            self.win_state = False

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        reward = 0
        movement, space_now, shift_now = action[0], action[1] == 1, action[2] == 1
        _, space_prev, shift_prev = self.last_action
        
        space_press = space_now and not space_prev
        shift_press = shift_now and not shift_prev

        # --- Cursor Movement ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        elif movement != 0:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE[1] - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE[0] - 1, self.cursor_pos[0] + 1)
            self.move_cooldown = 5

        # --- Actions ---
        if shift_press:
            self.selected_plant_idx = (self.selected_plant_idx + 1) % len(self.CROP_DATA)
            # sfx: cycle_sound
            pos = (self.SCREEN_SIZE[0] // 2, self.SCREEN_SIZE[1] - 30)
            self._add_particles(pos, 10, self.CROP_DATA[self.selected_plant_idx]['color'], 2, 15)

        if space_press:
            cx, cy = self.cursor_pos
            plot = self.plots[cy][cx]
            
            # --- Market Interaction ---
            if (cx, cy) == self.MARKET_POS:
                money_earned = 0
                items_sold = 0
                for crop_name, count in self.inventory.items():
                    if count > 0:
                        crop_data = next(c for c in self.CROP_DATA if c['name'] == crop_name)
                        money_earned += count * crop_data['sell_price']
                        items_sold += count
                        self.inventory[crop_name] = 0
                if money_earned > 0:
                    self.money += money_earned
                    reward += items_sold * 1.0  # +1 per crop sold
                    pos = self._iso_transform(cx, cy)
                    self._add_floating_text(f"+${money_earned}", pos, (255, 223, 0))
                    # sfx: cash_register
                    self._add_particles(pos, 30, (255, 223, 0), 3, 30)

            # --- Farm Plot Interaction ---
            else:
                crop_idx = plot['type']
                if crop_idx is not None:
                    # Harvest
                    crop_data = self.CROP_DATA[crop_idx]
                    if plot['growth'] >= crop_data['growth_time']:
                        self.inventory[crop_data['name']] += 1
                        plot['type'] = None
                        plot['growth'] = 0
                        reward += 0.1
                        pos = self._iso_transform(cx, cy)
                        self._add_floating_text(f"+1 {crop_data['name']}", pos, crop_data['color'])
                        # sfx: harvest_sound
                        self._add_particles(pos, 15, crop_data['color'], 2.5, 20)
                else:
                    # Plant
                    selected_crop = self.CROP_DATA[self.selected_plant_idx]
                    if self.money >= selected_crop['cost']:
                        self.money -= selected_crop['cost']
                        plot['type'] = self.selected_plant_idx
                        plot['growth'] = 0
                        pos = self._iso_transform(cx, cy)
                        self._add_floating_text(f"-${selected_crop['cost']}", pos, (255, 80, 80))
                        # sfx: plant_sound
                        self._add_particles(pos, 15, selected_crop['leaf_color'], 1.5, 20)
        
        self.last_action = [movement, space_now, shift_now]
        return reward

    def _update_plots(self):
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                plot = self.plots[y][x]
                if plot['type'] is not None:
                    crop_data = self.CROP_DATA[plot['type']]
                    if plot['growth'] < crop_data['growth_time']:
                        plot['growth'] += 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and crops
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                screen_pos = self._iso_transform(x, y)
                
                is_market = (x, y) == self.MARKET_POS
                tile_color = self.COLOR_MARKET if is_market else self.COLOR_GRID
                
                points = [
                    (screen_pos[0], screen_pos[1] - self.TILE_H // 2),
                    (screen_pos[0] + self.TILE_W // 2, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] + self.TILE_H // 2),
                    (screen_pos[0] - self.TILE_W // 2, screen_pos[1]),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, tile_color)
                pygame.gfxdraw.filled_polygon(self.screen, points, tile_color)

                if not is_market:
                    plot = self.plots[y][x]
                    if plot['type'] is not None:
                        self._render_crop(plot, screen_pos)
                    else:
                        dirt_points = [
                            (screen_pos[0], screen_pos[1] - self.TILE_H // 4),
                            (screen_pos[0] + self.TILE_W // 4, screen_pos[1]),
                            (screen_pos[0], screen_pos[1] + self.TILE_H // 4),
                            (screen_pos[0] - self.TILE_W // 4, screen_pos[1]),
                        ]
                        pygame.gfxdraw.filled_polygon(self.screen, dirt_points, self.COLOR_DIRT)
                else: # Market Stall drawing
                    pygame.draw.rect(self.screen, (120, 50, 50), (screen_pos[0]-10, screen_pos[1]-20, 20, 10))
                    pygame.draw.rect(self.screen, (220, 220, 200), (screen_pos[0]-12, screen_pos[1]-25, 24, 5))
                    self._render_text("$", screen_pos, self.FONT_M, (255,223,0), center=True)


        # Draw cursor
        cursor_screen_pos = self._iso_transform(self.cursor_pos[0], self.cursor_pos[1])
        cursor_points = [
            (cursor_screen_pos[0], cursor_screen_pos[1] - self.TILE_H // 2 - 2),
            (cursor_screen_pos[0] + self.TILE_W // 2 + 2, cursor_screen_pos[1]),
            (cursor_screen_pos[0], cursor_screen_pos[1] + self.TILE_H // 2 + 2),
            (cursor_screen_pos[0] - self.TILE_W // 2 - 2, cursor_screen_pos[1]),
        ]
        alpha = int(128 + 127 * math.sin(self.steps * 0.2))
        pygame.gfxdraw.aapolygon(self.screen, cursor_points, (*self.COLOR_CURSOR, alpha))
        
        # Draw particles and floating texts
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))
        for ft in self.floating_texts:
            self._render_text(ft['text'], ft['pos'], self.FONT_S, (*ft['color'], ft['alpha']), center=True)

    def _render_crop(self, plot, pos):
        crop_data = self.CROP_DATA[plot['type']]
        progress = plot['growth'] / crop_data['growth_time']
        
        if progress >= 1.0:
            # Pulsing glow for ready-to-harvest crops
            alpha = int(100 + 90 * math.sin(self.steps * 0.25))
            radius = int(self.TILE_W / 3)
            glow_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*crop_data['color'], alpha), (radius, radius), radius)
            self.screen.blit(glow_surf, (pos[0] - radius, pos[1] - radius - 5))

        # Render the crop itself, scaling with growth
        h = max(1, int(20 * progress))
        w = max(1, int(5 + 10 * progress))
        
        # Carrots: show leaves then orange top
        if crop_data['name'] == 'Carrots':
            leaf_h = int(h * 0.7)
            leaf_pos = (pos[0], pos[1] - leaf_h // 2 - 5)
            pygame.draw.ellipse(self.screen, crop_data['leaf_color'], (leaf_pos[0]-w//2, leaf_pos[1]-leaf_h//2, w, leaf_h))
            if progress > 0.5:
                carrot_h = int((progress - 0.5) * 2 * 10)
                pygame.draw.ellipse(self.screen, crop_data['color'], (pos[0]-w//3, pos[1]-5, w//1.5, carrot_h))
        # Corn: tall stalk
        elif crop_data['name'] == 'Corn':
            stalk_w = max(1, int(w * 0.4))
            pygame.draw.rect(self.screen, crop_data['color'], (pos[0]-stalk_w//2, pos[1]-h-5, stalk_w, h))
            if progress > 0.6:
                cob_size = int((progress - 0.6) * 4 * 5)
                pygame.draw.ellipse(self.screen, (255,223,0), (pos[0], pos[1]-h*0.7-5, cob_size, cob_size*1.5))
        # Wheat: simple vertical lines
        else:
            pygame.draw.line(self.screen, crop_data['color'], (pos[0], pos[1]-5), (pos[0], pos[1]-h-5), int(w*0.5))
            pygame.draw.line(self.screen, crop_data['color'], (pos[0]-3, pos[1]-5), (pos[0]-3, pos[1]-h*0.8-5), int(w*0.4))
            pygame.draw.line(self.screen, crop_data['color'], (pos[0]+3, pos[1]-5), (pos[0]+3, pos[1]-h*0.8-5), int(w*0.4))

    def _render_ui(self):
        # --- UI Backgrounds ---
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_SIZE[0], 40))
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, self.SCREEN_SIZE[1] - 70, self.SCREEN_SIZE[0], 70))
        
        # --- Top Bar: Money and Time ---
        self._render_text(f"$ {self.money}", (15, 20), self.FONT_M, self.COLOR_TEXT)
        time_left = max(0, (self.MAX_STEPS - self.steps) // 30)
        time_color = (255, 100, 100) if time_left < 30 else self.COLOR_TEXT
        self._render_text(f"{time_left}s", (self.SCREEN_SIZE[0] - 15, 20), self.FONT_M, time_color, align='right')

        # --- Bottom Bar: Inventory and Selected Plant ---
        # Inventory
        start_x = self.SCREEN_SIZE[0] // 2
        for i, (name, count) in enumerate(self.inventory.items()):
            crop_data = self.CROP_DATA[i]
            self._render_text(f"{name}: {count}", (start_x, self.SCREEN_SIZE[1] - 50 + i * 20), self.FONT_S, self.COLOR_TEXT, center=True)

        # Selected Plant
        selected_crop = self.CROP_DATA[self.selected_plant_idx]
        self._render_text("Planting:", (20, self.SCREEN_SIZE[1] - 55), self.FONT_S, self.COLOR_TEXT)
        self._render_text(selected_crop['name'], (20, self.SCREEN_SIZE[1] - 30), self.FONT_M, selected_crop['color'])

        # --- Game Over/Win Text ---
        if self.game_over:
            overlay = pygame.Surface(self.SCREEN_SIZE, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            if self.win_state:
                self._render_text("YOU WIN!", (self.SCREEN_SIZE[0]//2, self.SCREEN_SIZE[1]//2 - 20), self.FONT_L, (255, 223, 0), center=True)
                self._render_text(f"Final Money: ${self.money}", (self.SCREEN_SIZE[0]//2, self.SCREEN_SIZE[1]//2 + 40), self.FONT_M, self.COLOR_TEXT, center=True)
            else:
                self._render_text("TIME'S UP", (self.SCREEN_SIZE[0]//2, self.SCREEN_SIZE[1]//2 - 20), self.FONT_L, (200, 50, 50), center=True)
                self._render_text(f"Final Money: ${self.money}", (self.SCREEN_SIZE[0]//2, self.SCREEN_SIZE[1]//2 + 40), self.FONT_M, self.COLOR_TEXT, center=True)

    def _render_text(self, text, pos, font, color, center=False, align='left'):
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surf = font.render(text, True, color)
        
        text_rect = text_surf.get_rect()
        if center: text_rect.center = pos
        elif align == 'right': text_rect.topright = pos
        else: text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _iso_transform(self, x, y):
        return (
            int(self.ORIGIN_X + (x - y) * self.TILE_W / 2),
            int(self.ORIGIN_Y + (x + y) * self.TILE_H / 2)
        )
    
    def _add_particles(self, pos, count, color, speed_max, life_max):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, speed_max)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': random.randint(life_max // 2, life_max),
                'max_life': life_max,
                'size': random.uniform(2, 5),
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            p['size'] *= 0.97
            if p['life'] <= 0:
                self.particles.remove(p)
    
    def _add_floating_text(self, text, pos, color):
        self.floating_texts.append({
            'text': text,
            'pos': list(pos),
            'life': 45,
            'color': color,
            'alpha': 255,
        })

    def _update_floating_texts(self):
        for ft in self.floating_texts[:]:
            ft['pos'][1] -= 0.5
            ft['life'] -= 1
            ft['alpha'] = max(0, int(255 * (ft['life'] / 45)))
            if ft['life'] <= 0:
                self.floating_texts.remove(ft)

    def _get_info(self):
        return {
            "score": self.money,
            "steps": self.steps,
            "cumulative_reward": self.cumulative_reward,
            "inventory": self.inventory,
        }
        
    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_human = pygame.display.set_mode((env.SCREEN_SIZE[0], env.SCREEN_SIZE[1]))
    pygame.display.set_caption("Farming Simulator")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()
    pygame.quit()