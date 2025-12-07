
# Generated: 2025-08-28T06:32:17.962324
# Source Brief: brief_02959.md
# Brief Index: 2959

        
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
    """
    A fast-paced arcade farming simulator. Players must plant, grow, and sell
    crops to reach a target profit before time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the selector. "
        "Hold Space to plant the selected seed. "
        "Hold Shift to harvest a ready crop or sell all harvested crops at the market."
    )
    game_description = (
        "Fast-paced farming sim! Plant crops, watch them grow, and sell them at the market. "
        "Reach the profit goal of 1000 coins in 90 seconds. Prices fluctuate, so sell wisely!"
    )

    # Frame advance setting
    auto_advance = True

    # --- Game Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    
    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_UI_BG = (40, 50, 60)
    COLOR_GRID_BG = (101, 67, 33) # Soil
    COLOR_GRID_LINE = (81, 47, 13)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_GOLD = (255, 215, 0)
    COLOR_TIMER_WARN = (255, 100, 100)
    
    # Timing
    FPS = 30
    GAME_DURATION_SECONDS = 90
    
    # Gameplay
    WIN_CONDITION_COINS = 1000
    STARTING_COINS = 50
    
    # Farm Grid
    GRID_COLS, GRID_ROWS = 5, 3
    GRID_CELL_SIZE = 60
    GRID_PADDING = 10
    GRID_START_X = 30
    GRID_START_Y = 100
    
    # Market
    MARKET_X = GRID_START_X + (GRID_COLS * (GRID_CELL_SIZE + GRID_PADDING)) + 40
    MARKET_Y = GRID_START_Y
    MARKET_W = 180
    MARKET_H = GRID_ROWS * (GRID_CELL_SIZE + GRID_PADDING) - GRID_PADDING
    
    # Crop Data
    CROP_DATA = {
        'Carrot': {'growth_time': 10 * FPS, 'seed_cost': 5, 'base_price': 15, 'color': (255, 165, 0)},
        'Cabbage': {'growth_time': 20 * FPS, 'seed_cost': 15, 'base_price': 50, 'color': (144, 238, 144)},
        'Strawberry': {'growth_time': 30 * FPS, 'seed_cost': 30, 'base_price': 120, 'color': (220, 20, 60)},
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_main = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        self.font_title = pygame.font.Font(None, 48)
        self.font_floating = pygame.font.Font(None, 22)
        
        # Internal state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.coins = 0
        self.timer_frames = 0
        self.farm_grid = []
        self.cursor_pos = [0, 0] # [col, row]
        self.cursor_on_market = False
        self.crop_inventory = {}
        self.seed_inventory = {}
        self.market_prices = {}
        self.particles = []
        self.floating_texts = []
        self.selected_seed_idx = 0
        self.seed_types = list(self.CROP_DATA.keys())
        self.last_movement_time = 0

        # Run validation check
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0 # This is the episode's cumulative reward, not game coins
        self.game_over = False
        
        self.coins = self.STARTING_COINS
        self.timer_frames = self.GAME_DURATION_SECONDS * self.FPS
        
        self.farm_grid = [
            [{'state': 'empty', 'type': None, 'growth': 0} for _ in range(self.GRID_ROWS)]
            for _ in range(self.GRID_COLS)
        ]
        self.cursor_pos = [0, 0]
        self.cursor_on_market = False
        
        self.crop_inventory = {crop: 0 for crop in self.CROP_DATA}
        self.seed_inventory = {'Carrot': 10, 'Cabbage': 5, 'Strawberry': 2}
        
        self._update_market_prices()
        
        self.particles = []
        self.floating_texts = []
        
        self.selected_seed_idx = 0
        self.last_movement_time = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = -0.01 # Small penalty for existing, encourages action

        # --- Handle Input and Actions ---
        self._handle_movement(movement)
        
        if space_held:
            reward += self._action_plant()
            
        if shift_held:
            # Shift action is context-sensitive
            if self.cursor_on_market:
                reward += self._action_sell()
            else:
                reward += self._action_harvest()

        # --- Update Game State ---
        self._update_crops()
        self._update_particles()
        self._update_floating_texts()
        
        self.timer_frames -= 1
        if self.steps % (5 * self.FPS) == 0: # Prices change every 5 seconds
            self._update_market_prices()
        
        # --- Check Termination ---
        terminated = False
        if self.coins >= self.WIN_CONDITION_COINS:
            reward += 1000
            terminated = True
            self._add_floating_text("YOU WIN!", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_TEXT_GOLD, 150, 60)
        elif self.timer_frames <= 0:
            reward -= 100
            terminated = True
            self._add_floating_text("TIME UP!", (self.WIDTH // 2, self.HEIGHT // 2), self.COLOR_TIMER_WARN, 150, 60)
        
        self.game_over = terminated
        self.steps += 1
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        # Cooldown to prevent cursor from flying too fast
        if self.steps - self.last_movement_time < 3:
            return
        
        moved = False
        if movement == 1: # Up
            if not self.cursor_on_market: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            moved = True
        elif movement == 2: # Down
            if not self.cursor_on_market: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
            moved = True
        elif movement == 3: # Left
            if self.cursor_on_market:
                self.cursor_on_market = False
                self.cursor_pos[0] = self.GRID_COLS - 1
            else:
                self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            moved = True
        elif movement == 4: # Right
            if not self.cursor_on_market and self.cursor_pos[0] == self.GRID_COLS - 1:
                self.cursor_on_market = True
            elif not self.cursor_on_market:
                self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)
            moved = True
        
        if moved:
            self.last_movement_time = self.steps

    def _action_plant(self):
        if self.cursor_on_market: # Cannot plant on market
            # Cycle through seeds instead
            self.selected_seed_idx = (self.selected_seed_idx + 1) % len(self.seed_types)
            self.last_movement_time = self.steps + 5 # Add a small delay to prevent rapid cycling
            return 0
            
        col, row = self.cursor_pos
        plot = self.farm_grid[col][row]
        
        selected_seed = self.seed_types[self.selected_seed_idx]
        seed_cost = self.CROP_DATA[selected_seed]['seed_cost']
        
        if plot['state'] == 'empty' and self.seed_inventory[selected_seed] > 0 and self.coins >= seed_cost:
            plot['state'] = 'growing'
            plot['type'] = selected_seed
            plot['growth'] = 0
            
            self.coins -= seed_cost
            self.seed_inventory[selected_seed] -= 1
            
            # Visual feedback
            # # Sound: Plant seed
            pos = self._get_plot_center(col, row)
            self._add_particles(20, pos, self.CROP_DATA[selected_seed]['color'])
            self._add_floating_text(f"-{seed_cost}", (pos[0], pos[1] - 20), self.COLOR_TEXT_GOLD)
            return 5 # Reward for planting
        return -0.1 # Penalty for failed action

    def _action_harvest(self):
        if self.cursor_on_market: return -0.1
        
        col, row = self.cursor_pos
        plot = self.farm_grid[col][row]
        
        if plot['state'] == 'ready':
            crop_type = plot['type']
            self.crop_inventory[crop_type] += 1
            
            # Reset plot
            plot['state'] = 'empty'
            plot['type'] = None
            plot['growth'] = 0
            
            # Visual feedback
            # # Sound: Harvest
            pos = self._get_plot_center(col, row)
            self._add_particles(30, pos, self.CROP_DATA[crop_type]['color'], life=40, gravity=0.1, target=(self.WIDTH-100, 50))
            return 10 # Reward for harvesting
        return -0.1

    def _action_sell(self):
        total_sale = 0
        sold_anything = False
        for crop, count in self.crop_inventory.items():
            if count > 0:
                sale_value = count * self.market_prices[crop]
                total_sale += sale_value
                self.crop_inventory[crop] = 0
                sold_anything = True
        
        if sold_anything:
            self.coins += total_sale
            # Visual feedback
            # # Sound: Cha-ching!
            pos = (self.MARKET_X + self.MARKET_W // 2, self.MARKET_Y + self.MARKET_H // 2)
            self._add_floating_text(f"+{total_sale}", pos, self.COLOR_TEXT_GOLD, 60)
            return 25 # Reward for selling
        return -0.1

    def _update_crops(self):
        for col in range(self.GRID_COLS):
            for row in range(self.GRID_ROWS):
                plot = self.farm_grid[col][row]
                if plot['state'] == 'growing':
                    plot['growth'] += 1
                    growth_time = self.CROP_DATA[plot['type']]['growth_time']
                    if plot['growth'] >= growth_time:
                        plot['state'] = 'ready'
                        # # Sound: Crop ready
                        pos = self._get_plot_center(col, row)
                        self._add_particles(15, pos, (255, 255, 100), life=20, gravity=0)

    def _update_market_prices(self):
        for crop, data in self.CROP_DATA.items():
            fluctuation = self.np_random.integers(-5, 6)
            self.market_prices[crop] = max(1, data['base_price'] + fluctuation)
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid()
        self._render_market()
        self._render_cursor()
        self._render_particles()
        self._render_floating_texts()

    def _render_grid(self):
        for col in range(self.GRID_COLS):
            for row in range(self.GRID_ROWS):
                x = self.GRID_START_X + col * (self.GRID_CELL_SIZE + self.GRID_PADDING)
                y = self.GRID_START_Y + row * (self.GRID_CELL_SIZE + self.GRID_PADDING)
                
                # Draw plot background
                pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x, y, self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), border_radius=5)
                
                # Draw crop
                plot = self.farm_grid[col][row]
                if plot['state'] != 'empty':
                    crop_data = self.CROP_DATA[plot['type']]
                    growth_time = crop_data['growth_time']
                    
                    if plot['state'] == 'growing':
                        progress = plot['growth'] / growth_time
                        radius = int(max(2, (self.GRID_CELL_SIZE / 2 - 5) * progress))
                        pygame.draw.circle(self.screen, crop_data['color'], (x + self.GRID_CELL_SIZE // 2, y + self.GRID_CELL_SIZE // 2), radius)
                    elif plot['state'] == 'ready':
                        pulse = abs(math.sin(self.steps * 0.2)) * 4
                        radius = int(self.GRID_CELL_SIZE / 2 - 5 + pulse)
                        pygame.gfxdraw.filled_circle(self.screen, x + self.GRID_CELL_SIZE // 2, y + self.GRID_CELL_SIZE // 2, radius, crop_data['color'])
                        pygame.gfxdraw.aacircle(self.screen, x + self.GRID_CELL_SIZE // 2, y + self.GRID_CELL_SIZE // 2, radius, crop_data['color'])
    
    def _render_market(self):
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.MARKET_X, self.MARKET_Y, self.MARKET_W, self.MARKET_H), border_radius=8)
        
        title_surf = self.font_main.render("Market", True, self.COLOR_TEXT_GOLD)
        self.screen.blit(title_surf, (self.MARKET_X + (self.MARKET_W - title_surf.get_width()) // 2, self.MARKET_Y + 10))
        
        y_offset = self.MARKET_Y + 45
        for crop, price in self.market_prices.items():
            crop_surf = self.font_small.render(f"{crop}:", True, self.CROP_DATA[crop]['color'])
            price_surf = self.font_small.render(f"{price} coins", True, self.COLOR_TEXT)
            self.screen.blit(crop_surf, (self.MARKET_X + 15, y_offset))
            self.screen.blit(price_surf, (self.MARKET_X + self.MARKET_W - price_surf.get_width() - 15, y_offset))
            y_offset += 25

    def _render_cursor(self):
        alpha = 128 + int(127 * math.sin(self.steps * 0.3))
        color = (*self.COLOR_CURSOR, alpha)
        
        cursor_surface = pygame.Surface((self.GRID_CELL_SIZE, self.GRID_CELL_SIZE), pygame.SRCALPHA)
        if self.cursor_on_market:
            cursor_surface = pygame.Surface((self.MARKET_W, self.MARKET_H), pygame.SRCALPHA)
            pygame.draw.rect(cursor_surface, color, cursor_surface.get_rect(), 5, border_radius=8)
            self.screen.blit(cursor_surface, (self.MARKET_X, self.MARKET_Y))
        else:
            pygame.draw.rect(cursor_surface, color, cursor_surface.get_rect(), 5, border_radius=5)
            col, row = self.cursor_pos
            x = self.GRID_START_X + col * (self.GRID_CELL_SIZE + self.GRID_PADDING)
            y = self.GRID_START_Y + row * (self.GRID_CELL_SIZE + self.GRID_PADDING)
            self.screen.blit(cursor_surface, (x, y))

    def _render_ui(self):
        # Top Bar
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.WIDTH, 80))
        pygame.draw.line(self.screen, self.COLOR_BG, (0, 80), (self.WIDTH, 80), 2)
        
        # Coins
        coin_text = f"{self.coins} / {self.WIN_CONDITION_COINS}"
        coin_surf = self.font_title.render(coin_text, True, self.COLOR_TEXT_GOLD)
        self.screen.blit(coin_surf, (20, 20))
        
        # Timer
        time_left_sec = max(0, self.timer_frames / self.FPS)
        timer_text = f"{int(time_left_sec // 60):02}:{int(time_left_sec % 60):02}"
        timer_color = self.COLOR_TIMER_WARN if time_left_sec < 10 else self.COLOR_TEXT
        timer_surf = self.font_title.render(timer_text, True, timer_color)
        self.screen.blit(timer_surf, (self.WIDTH - timer_surf.get_width() - 20, 20))

        # Inventory (Crops)
        y_offset = self.MARKET_Y + self.MARKET_H - 100
        inv_title_surf = self.font_main.render("Harvested", True, self.COLOR_TEXT)
        self.screen.blit(inv_title_surf, (self.MARKET_X + (self.MARKET_W - inv_title_surf.get_width()) // 2, y_offset))
        y_offset += 30
        for crop, count in self.crop_inventory.items():
            if count > 0:
                crop_surf = self.font_small.render(f"{crop}: {count}", True, self.CROP_DATA[crop]['color'])
                self.screen.blit(crop_surf, (self.MARKET_X + 15, y_offset))
                y_offset += 20
        
        # Selected Seed
        selected_seed = self.seed_types[self.selected_seed_idx]
        seed_data = self.CROP_DATA[selected_seed]
        seed_count = self.seed_inventory[selected_seed]
        seed_text = f"Planting: {selected_seed} ({seed_count} left)"
        seed_surf = self.font_main.render(seed_text, True, seed_data['color'])
        self.screen.blit(seed_surf, (self.GRID_START_X, self.GRID_START_Y + self.GRID_ROWS * (self.GRID_CELL_SIZE + self.GRID_PADDING) + 10))

    def _get_info(self):
        return {
            "score": self.coins,
            "steps": self.steps,
            "time_left": max(0, self.timer_frames / self.FPS),
        }

    # --- Visual Effects ---
    def _add_particles(self, count, pos, color, life=30, speed=3, gravity=0.05, target=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_x = math.cos(angle) * speed * self.np_random.uniform(0.5, 1.5)
            vel_y = math.sin(angle) * speed * self.np_random.uniform(0.5, 1.5)
            self.particles.append({'pos': list(pos), 'vel': [vel_x, vel_y], 'life': life, 'color': color, 'gravity': gravity, 'target': target})

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                continue
            
            if p['target']: # Homing particles
                target_vec = [p['target'][0] - p['pos'][0], p['target'][1] - p['pos'][1]]
                dist = math.hypot(*target_vec)
                if dist > 1:
                    target_vec = [v / dist for v in target_vec]
                p['vel'][0] = p['vel'][0] * 0.9 + target_vec[0] * 1.5
                p['vel'][1] = p['vel'][1] * 0.9 + target_vec[1] * 1.5
            else: # Standard particles
                p['vel'][1] += p['gravity']
                
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(3 * (p['life'] / 30)))
            pygame.draw.rect(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1]), size, size))

    def _add_floating_text(self, text, pos, color, life=45, size=22):
        font = pygame.font.Font(None, size)
        self.floating_texts.append({'text': text, 'pos': list(pos), 'life': life, 'max_life': life, 'color': color, 'font': font})

    def _update_floating_texts(self):
        for ft in self.floating_texts[:]:
            ft['life'] -= 1
            if ft['life'] <= 0:
                self.floating_texts.remove(ft)
                continue
            ft['pos'][1] -= 0.5 # Float upwards

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            alpha = int(255 * (ft['life'] / ft['max_life']))
            text_surf = ft['font'].render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            pos = (ft['pos'][0] - text_surf.get_width() // 2, ft['pos'][1] - text_surf.get_height() // 2)
            self.screen.blit(text_surf, pos)
    
    def _get_plot_center(self, col, row):
        x = self.GRID_START_X + col * (self.GRID_CELL_SIZE + self.GRID_PADDING) + self.GRID_CELL_SIZE // 2
        y = self.GRID_START_Y + row * (self.GRID_CELL_SIZE + self.GRID_PADDING) + self.GRID_CELL_SIZE // 2
        return (x, y)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Farming Simulator")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("      MANUAL PLAY TEST")
    print("="*30)
    print(GameEnv.game_description)
    print("\n" + GameEnv.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Action mapping for human keyboard input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Final Coins: {info['score']}. Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting
            
        clock.tick(GameEnv.FPS)
        
    env.close()