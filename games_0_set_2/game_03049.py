
# Generated: 2025-08-28T06:51:38.937221
# Source Brief: brief_03049.md
# Brief Index: 3049

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to plant a seed. Shift to harvest a crop and sell your stock."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage your farm to earn 1000 gold. Plant seeds, wait for them to grow, "
        "then harvest and sell them. Beat the clock to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 12, 8
    CELL_SIZE = 32
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_ROWS * CELL_SIZE) // 2 + 20

    MAX_TIMER = 5000  # in frames (166s at 30fps)
    WIN_GOLD = 1000
    
    INITIAL_SEEDS = 10
    CROP_GROWTH_TIME = 240 # frames (8s at 30fps)
    CROP_SELL_PRICE = 25

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_SOIL = (89, 61, 43)
    COLOR_GRID = (111, 78, 55)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_CROP_GROWING = (80, 200, 120)
    COLOR_CROP_READY = (255, 215, 0)
    COLOR_UI_BG = (0, 0, 0, 150)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_WIN_TEXT = (124, 252, 0)
    COLOR_LOSE_TEXT = (255, 69, 0)

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
        
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        self.farm_plots = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0, 0]
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.gold = 0
        self.seeds = self.INITIAL_SEEDS
        self.harvested_crops = 0
        self.timer = self.MAX_TIMER
        self.game_over = False
        self.win_state = False
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        plot_start_x = self.GRID_X_OFFSET + self.CELL_SIZE / 2
        plot_start_y = self.GRID_Y_OFFSET + self.CELL_SIZE / 2
        self.visual_cursor_pos = [
            plot_start_x + self.cursor_pos[0] * self.CELL_SIZE,
            plot_start_y + self.cursor_pos[1] * self.CELL_SIZE
        ]

        self.farm_plots = [
            {'state': 'empty', 'growth': 0} 
            for _ in range(self.GRID_COLS * self.GRID_ROWS)
        ]
        
        self.particles = []

        self.last_space_press = False
        self.last_shift_press = False
        self.steps = 0 # This will be frame count
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            self.timer -= 1
            self.steps += 1
            
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- Handle Input (edge-triggered) ---
            space_press = space_held and not self.last_space_press
            shift_press = shift_held and not self.last_shift_press
            
            self.last_space_press = space_held
            self.last_shift_press = shift_held

            if movement != 0:
                self._move_cursor(movement)
            
            if space_press:
                reward += self._plant()
            
            if shift_press:
                reward += self._harvest_and_sell()

            # Idle penalty
            if movement == 0 and not space_held and not shift_held:
                reward -= 0.01

            # --- Update Game State ---
            self._update_crops()
            self._update_visuals()

        self._update_particles()
        
        # --- Check Termination ---
        terminated = False
        if not self.game_over:
            if self.timer <= 0:
                self.game_over = True
                self.win_state = False
                terminated = True
            elif self.gold >= self.WIN_GOLD:
                self.game_over = True
                self.win_state = True
                terminated = True
                reward += 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, direction):
        if direction == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif direction == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif direction == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif direction == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

    def _plant(self):
        plot_idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
        plot = self.farm_plots[plot_idx]
        
        if self.seeds > 0 and plot['state'] == 'empty':
            plot['state'] = 'growing'
            plot['growth'] = 0
            self.seeds -= 1
            # sfx: plant_seed
            pos = self._get_cursor_pixel_pos()
            self._create_particles(pos, 'plant')
            return 0.1 # Small reward for planting
        return 0

    def _harvest_and_sell(self):
        step_reward = 0
        
        # 1. Harvest at cursor
        plot_idx = self.cursor_pos[1] * self.GRID_COLS + self.cursor_pos[0]
        plot = self.farm_plots[plot_idx]
        
        if plot['state'] == 'ready':
            plot['state'] = 'empty'
            plot['growth'] = 0
            self.harvested_crops += 1
            step_reward += 1.0 # Harvest reward
            # sfx: harvest
            pos = self._get_cursor_pixel_pos()
            self._create_particles(pos, 'harvest')

        # 2. Sell all harvested crops
        if self.harvested_crops > 0:
            sold_amount = self.harvested_crops
            self.gold += sold_amount * self.CROP_SELL_PRICE
            self.harvested_crops = 0
            step_reward += 5.0 # Sell reward
            # sfx: cha-ching
            ui_gold_pos = (self.SCREEN_WIDTH - 70, 25)
            self._create_particles(ui_gold_pos, 'sell', num=sold_amount)
            
        return step_reward

    def _update_crops(self):
        for plot in self.farm_plots:
            if plot['state'] == 'growing':
                plot['growth'] += 1
                if plot['growth'] >= self.CROP_GROWTH_TIME:
                    plot['state'] = 'ready'
                    # sfx: crop_ready_ding

    def _update_visuals(self):
        target_pos = self._get_cursor_pixel_pos()
        self.visual_cursor_pos[0] += (target_pos[0] - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (target_pos[1] - self.visual_cursor_pos[1]) * 0.4

    def _create_particles(self, pos, p_type, num=10):
        for _ in range(num):
            if p_type == 'plant':
                p = {
                    'pos': list(pos),
                    'vel': [self.np_random.uniform(-1, 1), self.np_random.uniform(-2, -0.5)],
                    'life': self.np_random.integers(15, 25),
                    'color': (139, 69, 19),
                    'radius': self.np_random.uniform(2, 4)
                }
            elif p_type == 'harvest':
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                p = {
                    'pos': list(pos),
                    'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                    'life': self.np_random.integers(20, 30),
                    'color': self.COLOR_CROP_READY,
                    'radius': self.np_random.uniform(3, 6)
                }
            elif p_type == 'sell':
                p = {
                    'pos': [self.SCREEN_WIDTH - 200, self.np_random.uniform(50, 150)],
                    'vel': [self.np_random.uniform(2, 4), self.np_random.uniform(-1, 1)],
                    'life': self.np_random.integers(40, 60),
                    'color': self.COLOR_CROP_READY,
                    'radius': self.np_random.uniform(5, 8)
                }
            self.particles.append(p)

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
        self.particles = [p for p in self.particles if p['life'] > 0 and p['radius'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw farm plots
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                x = self.GRID_X_OFFSET + col * self.CELL_SIZE
                y = self.GRID_Y_OFFSET + row * self.CELL_SIZE
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw crops
        for i, plot in enumerate(self.farm_plots):
            if plot['state'] != 'empty':
                col = i % self.GRID_COLS
                row = i // self.GRID_COLS
                x = self.GRID_X_OFFSET + col * self.CELL_SIZE + self.CELL_SIZE // 2
                y = self.GRID_Y_OFFSET + row * self.CELL_SIZE + self.CELL_SIZE // 2
                
                if plot['state'] == 'growing':
                    progress = plot['growth'] / self.CROP_GROWTH_TIME
                    radius = int(2 + progress * (self.CELL_SIZE / 2 - 6))
                    color = self.COLOR_CROP_GROWING
                    pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
                    pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)
                elif plot['state'] == 'ready':
                    radius = int(self.CELL_SIZE / 2 - 4)
                    color = self.COLOR_CROP_READY
                    pulse = abs(math.sin(self.timer * 0.1)) * 3
                    pygame.gfxdraw.aacircle(self.screen, x, y, int(radius + pulse), color)
                    pygame.gfxdraw.filled_circle(self.screen, x, y, int(radius + pulse), color)
                    pygame.draw.line(self.screen, self.COLOR_CROP_GROWING, (x,y-radius), (x, y-radius-5), 2)
        
        # Draw cursor
        x, y = self.visual_cursor_pos
        size = self.CELL_SIZE * 1.2
        half_size = size / 2
        rect = pygame.Rect(x - half_size, y - half_size, size, size)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        alpha = 90 + abs(math.sin(self.timer * 0.05)) * 30
        pygame.draw.rect(surf, self.COLOR_CURSOR + (int(alpha),), surf.get_rect(), border_radius=5)
        pygame.draw.rect(surf, self.COLOR_CURSOR, surf.get_rect(), 2, border_radius=5)
        self.screen.blit(surf, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))
        
        timer_text = f"Time: {max(0, self.timer // 30):03}"
        self._draw_text(timer_text, (10, 15), self.font_small, self.COLOR_UI_TEXT)

        gold_text = f"Gold: {self.gold}/{self.WIN_GOLD}"
        self._draw_text(gold_text, (self.SCREEN_WIDTH - 180, 15), self.font_small, self.COLOR_CROP_READY)

        inventory_panel = pygame.Surface((200, 60), pygame.SRCALPHA)
        inventory_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(inventory_panel, (10, self.SCREEN_HEIGHT - 70))

        seeds_text = f"Seeds: {self.seeds}"
        self._draw_text(seeds_text, (20, self.SCREEN_HEIGHT - 60), self.font_small, self.COLOR_UI_TEXT)
        
        harvested_text = f"Harvested: {self.harvested_crops}"
        self._draw_text(harvested_text, (20, self.SCREEN_HEIGHT - 40), self.font_small, self.COLOR_UI_TEXT)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        if self.win_state:
            text = "YOU WIN!"
            color = self.COLOR_WIN_TEXT
        else:
            text = "TIME'S UP!"
            color = self.COLOR_LOSE_TEXT
            
        text_surf = self.font_large.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _get_info(self):
        return {
            "score": self.gold,
            "steps": self.steps,
        }

    def _get_cursor_pixel_pos(self):
        x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return (x, y)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment for human play
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Farming Sim")
    
    running = True
    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
            
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert observation from (H, W, C) to (W, H, C) for Pygame display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        env.clock.tick(30)
        
    env.close()