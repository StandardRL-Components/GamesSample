
# Generated: 2025-08-28T04:53:53.182161
# Source Brief: brief_05404.md
# Brief Index: 5404

        
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
        "Controls: ↑↓←→ to select a plot. Space to plant/harvest. Hold Shift to sell crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate a thriving farm by planting, harvesting, and selling crops to reach the gold target before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Grid settings
        self.GRID_ROWS = 4
        self.GRID_COLS = 5
        self.PLOT_SIZE = 60
        self.PLOT_PADDING = 15
        
        # Timing
        self.MAX_TIME_SECONDS = 120
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS
        self.GROW_TIME_SECONDS = 8
        self.GROW_TIME_STEPS = self.GROW_TIME_SECONDS * self.FPS
        
        # Economy
        self.STARTING_GOLD = 0
        self.WIN_GOLD = 500
        self.CROP_PRICE = 10
        
        # Rewards
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -100.0
        self.REWARD_HARVEST = 0.1
        self.REWARD_SELL_BASE = 1.0
        self.REWARD_STEP_PENALTY = -0.001

        # Colors
        self.COLOR_BG = (40, 50, 40)
        self.COLOR_FARM_AREA = (86, 112, 71)
        self.COLOR_PLOT_EMPTY = (92, 64, 51)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_SEEDLING = (144, 238, 144)
        self.COLOR_GROWING = (50, 205, 50)
        self.COLOR_RIPE = (255, 165, 0)
        self.COLOR_RIPE_GLOW = (255, 215, 0)
        self.COLOR_STALL = (139, 69, 19)
        self.COLOR_STALL_ROOF_1 = (255, 255, 255)
        self.COLOR_STALL_ROOF_2 = (220, 20, 60)
        self.COLOR_TEXT = (255, 255, 240)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_TIMER_DANGER = (255, 60, 60)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # State variables
        self.farm_grid = []
        self.cursor_pos = [0, 0]
        self.gold = 0
        self.harvested_crops = 0
        self.steps = 0
        self.time_remaining = 0
        self.game_over = False
        self.win = False
        self.last_space_held = False
        self.last_shift_held = False
        self.floating_texts = []

        self.grid_start_x = (self.WIDTH - (self.GRID_COLS * (self.PLOT_SIZE + self.PLOT_PADDING) - self.PLOT_PADDING)) // 2
        self.grid_start_y = (self.HEIGHT - (self.GRID_ROWS * (self.PLOT_SIZE + self.PLOT_PADDING) - self.PLOT_PADDING)) // 2 + 20

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.gold = self.STARTING_GOLD
        self.harvested_crops = 0
        self.time_remaining = self.MAX_STEPS
        self.game_over = False
        self.win = False
        self.cursor_pos = [0, 0]
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True # Prevent action on first frame
        self.floating_texts = []
        
        self.farm_grid = []
        for r in range(self.GRID_ROWS):
            row = []
            for c in range(self.GRID_COLS):
                row.append({'state': 'empty', 'growth': 0})
            self.farm_grid.append(row)
            
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = self.REWARD_STEP_PENALTY
        self.game_over = self.time_remaining <= 0 or self.gold >= self.WIN_GOLD
        
        if not self.game_over:
            # Unpack actions
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            space_pressed = space_held and not self.last_space_held
            shift_pressed = shift_held and not self.last_shift_held

            # 1. Update game logic based on player actions
            if movement == 1: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Up
            elif movement == 2: self.cursor_pos[0] = min(self.GRID_ROWS - 1, self.cursor_pos[0] + 1) # Down
            elif movement == 3: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Left
            elif movement == 4: self.cursor_pos[1] = min(self.GRID_COLS - 1, self.cursor_pos[1] + 1) # Right

            plot = self.farm_grid[self.cursor_pos[0]][self.cursor_pos[1]]
            if space_pressed:
                if plot['state'] == 'empty':
                    # Plant a seed
                    plot['state'] = 'growing'
                    plot['growth'] = self.GROW_TIME_STEPS
                    # sfx: plant_seed.wav
                elif plot['state'] == 'ripe':
                    # Harvest a crop
                    plot['state'] = 'empty'
                    plot['growth'] = 0
                    self.harvested_crops += 1
                    reward += self.REWARD_HARVEST
                    # sfx: harvest_pop.wav

            if shift_pressed and self.harvested_crops > 0:
                # Sell all harvested crops
                sold_amount = self.harvested_crops
                gold_earned = sold_amount * self.CROP_PRICE
                self.gold += gold_earned
                reward += self.REWARD_SELL_BASE * sold_amount
                self.harvested_crops = 0
                self._add_floating_text(f"+{gold_earned} G", (530, 350), self.COLOR_GOLD)
                # sfx: cash_register.wav

            # 2. Update time-based game logic
            self.time_remaining -= 1
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    p = self.farm_grid[r][c]
                    if p['state'] == 'growing':
                        p['growth'] -= 1
                        if p['growth'] <= 0:
                            p['state'] = 'ripe'
                            # sfx: crop_ready.wav
            
            self.last_space_held = space_held
            self.last_shift_held = shift_held
        
        # 3. Update animations
        self._update_floating_texts()

        # 4. Check for termination
        terminated = False
        if self.gold >= self.WIN_GOLD and not self.game_over:
            reward += self.REWARD_WIN
            terminated = True
            self.win = True
            self.game_over = True
        elif self.time_remaining <= 0 and not self.game_over:
            reward += self.REWARD_LOSE
            terminated = True
            self.win = False
            self.game_over = True

        self.steps += 1
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.gold, "steps": self.steps}

    def _render_game(self):
        # Draw farm area background
        farm_area_rect = pygame.Rect(
            self.grid_start_x - 30, self.grid_start_y - 30,
            self.GRID_COLS * (self.PLOT_SIZE + self.PLOT_PADDING) + 50,
            self.GRID_ROWS * (self.PLOT_SIZE + self.PLOT_PADDING) + 50
        )
        pygame.draw.rect(self.screen, self.COLOR_FARM_AREA, farm_area_rect, border_radius=15)
        
        # Draw plots and crops
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                plot_x = self.grid_start_x + c * (self.PLOT_SIZE + self.PLOT_PADDING)
                plot_y = self.grid_start_y + r * (self.PLOT_SIZE + self.PLOT_PADDING)
                plot_rect = pygame.Rect(plot_x, plot_y, self.PLOT_SIZE, self.PLOT_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, plot_rect, border_radius=5)
                
                plot = self.farm_grid[r][c]
                center_x, center_y = plot_rect.center
                
                if plot['state'] == 'growing':
                    progress = 1 - (plot['growth'] / self.GROW_TIME_STEPS)
                    radius = int(5 + (self.PLOT_SIZE * 0.3) * progress)
                    color = self.COLOR_SEEDLING if progress < 0.5 else self.COLOR_GROWING
                    pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
                elif plot['state'] == 'ripe':
                    radius = int(self.PLOT_SIZE * 0.4)
                    glow_radius = int(radius * (1.2 + 0.1 * math.sin(self.steps * 0.2)))
                    # Draw glow effect using anti-aliased circles
                    glow_alpha = int(90 + 30 * math.sin(self.steps * 0.2))
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, glow_radius, (*self.COLOR_RIPE_GLOW, glow_alpha))
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_RIPE)
        
        # Draw cursor
        cursor_x = self.grid_start_x + self.cursor_pos[1] * (self.PLOT_SIZE + self.PLOT_PADDING) - 5
        cursor_y = self.grid_start_y + self.cursor_pos[0] * (self.PLOT_SIZE + self.PLOT_PADDING) - 5
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.PLOT_SIZE + 10, self.PLOT_SIZE + 10)
        alpha = int(150 + 100 * math.sin(self.steps * 0.3))
        cursor_surface = pygame.Surface(cursor_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, (*self.COLOR_CURSOR, alpha), cursor_surface.get_rect(), 5, border_radius=8)
        self.screen.blit(cursor_surface, cursor_rect.topleft)

        # Draw market stall
        stall_rect = pygame.Rect(500, 280, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_STALL, stall_rect)
        for i in range(6):
            color = self.COLOR_STALL_ROOF_1 if i % 2 == 0 else self.COLOR_STALL_ROOF_2
            pygame.draw.line(self.screen, color, (500 + i*20, 280), (520 + i*20, 260), 3)
        pygame.draw.line(self.screen, (0,0,0), (500, 280), (620, 280), 3)

    def _render_ui(self):
        # Gold display
        pygame.gfxdraw.filled_circle(self.screen, 30, 30, 15, self.COLOR_GOLD)
        pygame.gfxdraw.aacircle(self.screen, 30, 30, 15, self.COLOR_TEXT)
        gold_text = self.font_medium.render(f"{self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (55, 15))

        # Timer display
        time_sec = self.time_remaining // self.FPS
        timer_color = self.COLOR_TEXT if time_sec > 10 else self.COLOR_TIMER_DANGER
        timer_text = self.font_medium.render(f"Time: {time_sec}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 20, 15))

        # Harvested crops display
        pygame.gfxdraw.filled_circle(self.screen, 530, 320, 10, self.COLOR_RIPE)
        harvest_text = self.font_medium.render(f"x {self.harvested_crops}", True, self.COLOR_TEXT)
        self.screen.blit(harvest_text, (550, 305))

        # Floating texts
        for ft in self.floating_texts:
            ft_surf = self.font_small.render(ft['text'], True, ft['color'])
            ft_surf.set_alpha(ft['alpha'])
            self.screen.blit(ft_surf, ft['pos'])

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win else "TIME'S UP!"
        color = self.COLOR_GOLD if self.win else self.COLOR_TIMER_DANGER
        
        text_surf = self.font_large.render(message, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
        self.screen.blit(text_surf, text_rect)
        
        score_surf = self.font_medium.render(f"Final Gold: {self.gold}", True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
        self.screen.blit(score_surf, score_rect)

    def _add_floating_text(self, text, pos, color):
        self.floating_texts.append({'text': text, 'pos': list(pos), 'color': color, 'life': self.FPS * 1.5})

    def _update_floating_texts(self):
        for ft in self.floating_texts[:]:
            ft['life'] -= 1
            ft['pos'][1] -= 0.5
            ft['alpha'] = max(0, int(255 * (ft['life'] / (self.FPS * 1.5))))
            if ft['life'] <= 0:
                self.floating_texts.remove(ft)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Farming Game")
    clock = pygame.time.Clock()

    print(GameEnv.user_guide)

    while not terminated:
        # --- Action gathering for human play ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering for human play ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling and clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()