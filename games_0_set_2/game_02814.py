
# Generated: 2025-08-27T21:31:04.656971
# Source Brief: brief_02814.md
# Brief Index: 2814

        
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
        "Controls: Arrows to move cursor. Space to plant Wheat. Shift to harvest a ripe crop and sell all harvested goods."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate crops, harvest produce, and sell goods to become the ultimate farm tycoon in this fast-paced grid-based farming simulator."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Game Constants ---
    # Colors
    COLOR_BG = (139, 172, 15)
    COLOR_SOIL = (92, 53, 15)
    COLOR_SOIL_GRID = (82, 43, 5)
    COLOR_CURSOR = (255, 255, 255, 100) # White with alpha
    COLOR_UI_BG = (48, 98, 48, 200)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_TIMER_BAR_BG = (87, 87, 87)
    COLOR_TIMER_BAR_FG = (255, 215, 0)
    COLOR_WIN = (255, 215, 0)
    COLOR_LOSE = (200, 0, 0)
    
    # Game parameters
    GRID_COLS, GRID_ROWS = 12, 8
    CELL_SIZE = 40
    GRID_MARGIN_X = (640 - GRID_COLS * CELL_SIZE) // 2
    GRID_MARGIN_Y = (400 - GRID_ROWS * CELL_SIZE) // 2 + 30
    
    WIN_SCORE = 1000
    MAX_STEPS = 6000
    INITIAL_SEEDS = 10
    INITIAL_COINS = 0

    CROP_DATA = {
        'wheat': {'grow_time': 100, 'value': 10, 'color_seed': (139, 69, 19), 'color_ripe': (245, 222, 179)},
        'carrot': {'grow_time': 200, 'value': 25, 'color_seed': (255, 140, 0), 'color_ripe': (255, 69, 0)},
    }
    
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
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Internal state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.timer = None
        self.game_over = None
        self.win_condition = None
        self.harvested_goods = None
        self.seeds = None
        self.particles = []
        self.rng = np.random.default_rng()
        
        # Initialize state variables
        self.reset()

        # self.validate_implementation() # Optional: Call to self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.score = self.INITIAL_COINS
        self.steps = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        self.win_condition = None # 'win' or 'lose'
        self.harvested_goods = {crop: 0 for crop in self.CROP_DATA}
        self.seeds = {'wheat': self.INITIAL_SEEDS}
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        plant_action = action[1] == 1  # Space button
        harvest_sell_action = action[2] == 1  # Shift button

        # --- Update Game Logic ---
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # 2. Handle planting
        if plant_action:
            # For now, space always plants wheat
            reward += self._plant_crop('wheat')

        # 3. Handle harvesting and selling
        if harvest_sell_action:
            harvest_reward = self._harvest_crop()
            sell_reward, _ = self._sell_goods()
            reward += harvest_reward + sell_reward

        # 4. Update crop growth
        self._update_crops()
        
        # 5. Update particles
        self._update_particles()
        
        # 6. Calculate step reward
        if movement == 0 and not plant_action and not harvest_sell_action:
            reward -= 0.01  # Penalty for inaction
        
        # 7. Check for termination
        terminated = self._check_termination()
        if terminated:
            if self.win_condition == 'win':
                reward += 100
            else:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _plant_crop(self, crop_type):
        x, y = self.cursor_pos
        if self.grid[y][x] is None and self.seeds.get(crop_type, 0) > 0:
            self.seeds[crop_type] -= 1
            self.grid[y][x] = {'type': crop_type, 'growth': 0}
            # sfx: plant_seed
            self._create_particles(self.GRID_MARGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                                   self.GRID_MARGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2,
                                   5, (139, 69, 19))
            return 0.05 # Small reward for planting
        return 0

    def _harvest_crop(self):
        x, y = self.cursor_pos
        crop = self.grid[y][x]
        if crop:
            crop_info = self.CROP_DATA[crop['type']]
            if crop['growth'] >= crop_info['grow_time']:
                self.grid[y][x] = None
                self.harvested_goods[crop['type']] += 1
                # sfx: harvest_pop
                self._create_particles(self.GRID_MARGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                                       self.GRID_MARGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2,
                                       10, crop_info['color_ripe'])
                return 0.1
        return 0

    def _sell_goods(self):
        total_value = 0
        items_sold = 0
        for crop_type, quantity in self.harvested_goods.items():
            if quantity > 0:
                value = self.CROP_DATA[crop_type]['value'] * quantity
                total_value += value
                items_sold += quantity
                self.harvested_goods[crop_type] = 0
        
        if total_value > 0:
            self.score += total_value
            # sfx: cash_register
            self._create_particles(85, 25, 20, self.COLOR_WIN, count=total_value)
            return 1.0, items_sold
        return 0, 0
        
    def _update_crops(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                crop = self.grid[y][x]
                if crop:
                    crop_info = self.CROP_DATA[crop['type']]
                    if crop['growth'] < crop_info['grow_time']:
                        crop['growth'] += 1

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            self.game_over = True
            self.win_condition = 'win'
            return True
        if self.timer <= 0:
            self.game_over = True
            self.win_condition = 'lose'
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_crops()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
            "harvested": self.harvested_goods,
            "seeds": self.seeds,
        }

    def _render_grid(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                rect = pygame.Rect(self.GRID_MARGIN_X + x * self.CELL_SIZE,
                                   self.GRID_MARGIN_Y + y * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                pygame.draw.rect(self.screen, self.COLOR_SOIL_GRID, rect, 1)

    def _render_crops(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                crop = self.grid[y][x]
                if crop:
                    crop_info = self.CROP_DATA[crop['type']]
                    progress = min(1.0, crop['growth'] / crop_info['grow_time'])
                    
                    center_x = self.GRID_MARGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
                    center_y = self.GRID_MARGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                    
                    # Interpolate color from seed to ripe
                    start_color = crop_info['color_seed']
                    end_color = crop_info['color_ripe']
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * progress)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * progress)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * progress)
                    color = (r,g,b)

                    # Size grows with progress
                    min_radius = 3
                    max_radius = self.CELL_SIZE // 2 - 4
                    radius = int(min_radius + (max_radius - min_radius) * progress)

                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
                    
                    # Add a sparkle if ripe
                    if progress >= 1.0:
                        sparkle_color = (255, 255, 100)
                        angle = (self.steps % 30) * (2 * math.pi / 30)
                        sx = int(center_x + (max_radius) * math.cos(angle))
                        sy = int(center_y + (max_radius) * math.sin(angle))
                        pygame.draw.circle(self.screen, sparkle_color, (sx, sy), 2)


    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(self.GRID_MARGIN_X + x * self.CELL_SIZE,
                           self.GRID_MARGIN_Y + y * self.CELL_SIZE,
                           self.CELL_SIZE, self.CELL_SIZE)
        
        # Use a surface with alpha for transparency
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, rect.topleft)
        pygame.draw.rect(self.screen, (255,255,255), rect, 2)


    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((640, 50), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Score
        score_text = self.font_small.render(f"💰 COINS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 5))
        
        # Seeds
        seed_text = self.font_small.render(f"🌾 SEEDS: {self.seeds.get('wheat', 0)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(seed_text, (10, 25))

        # Harvested Goods
        harvested_str = "🧺 HELD: " + ", ".join([f"{k[:1].upper()}:{v}" for k, v in self.harvested_goods.items() if v > 0])
        harvest_text = self.font_small.render(harvested_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(harvest_text, (200, 5))

        # Timer Bar
        timer_pct = self.timer / self.MAX_STEPS
        bar_width = 200
        bar_height = 15
        bar_x = 640 - bar_width - 10
        bar_y = 15
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_FG, (bar_x, bar_y, int(bar_width * timer_pct), bar_height))
        timer_text = self.font_small.render("TIME", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (bar_x - 50, 13))

    def _render_game_over(self):
        s = pygame.Surface((640, 400), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))

        if self.win_condition == 'win':
            text_str = "VICTORY!"
            color = self.COLOR_WIN
        else:
            text_str = "TIME UP!"
            color = self.COLOR_LOSE
            
        text = self.font_large.render(text_str, True, color)
        text_rect = text.get_rect(center=(640 // 2, 400 // 2))
        self.screen.blit(text, text_rect)
        
    def _create_particles(self, x, y, size, color, count=10):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            life = self.rng.integers(15, 30)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color
            })
            
    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _render_particles(self):
        for p in self.particles:
            life_pct = p['life'] / p['max_life']
            alpha = int(255 * life_pct)
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            # Use a surface for alpha blending
            s = pygame.Surface((6,6), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (3,3), int(3 * life_pct))
            self.screen.blit(s, (pos[0]-3, pos[1]-3))

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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    # To run validation
    # env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Farming Tycoon")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Default action is NO-OP
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        # The observation is (H, W, C), but pygame wants (W, H) surface
        # The _get_observation returns (H, W, C) from a (W, H) surface, so it's already correct
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()

        # Since auto_advance is False, we need a small delay for human playability
        clock.tick(15) # Limit to 15 actions per second for human play

    env.close()