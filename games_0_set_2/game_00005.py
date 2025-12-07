
# Generated: 2025-08-27T16:17:57.294262
# Source Brief: brief_00005.md
# Brief Index: 5

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to plant a seed or harvest a ready crop. Hold Shift to sell your harvested produce."
    )

    # Short, user-facing description of the game
    game_description = (
        "A fast-paced arcade farming simulator. Plant seeds, watch them grow, and harvest them for profit. Sell your crops to reach the coin goal before time runs out!"
    )

    # Frames auto-advance at a fixed rate
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Simulation steps per second
        self.MAX_STEPS = 5000 # ~166 seconds
        self.WIN_COINS = 1000
        
        # Grid dimensions
        self.GRID_SIZE = (12, 8)
        self.PLOT_SIZE = (40, 40)
        self.GRID_ORIGIN = (
            (self.WIDTH - self.GRID_SIZE[0] * self.PLOT_SIZE[0]) // 2,
            (self.HEIGHT - self.GRID_SIZE[1] * self.PLOT_SIZE[1]) // 2 + 20
        )
        
        # Crop properties
        self.GROWTH_TIME = 150 # steps (5 seconds at 30fps)
        self.CROP_VALUE = 5 # coins per crop
        
        # Colors
        self.COLOR_BG = (20, 30, 25)
        self.COLOR_PLOT_EMPTY = (87, 56, 36)
        self.COLOR_PLOT_PLANTED = (66, 42, 27)
        self.COLOR_CROP_START = (40, 80, 30)
        self.COLOR_CROP_END = (140, 220, 50)
        self.COLOR_CROP_READY = (255, 220, 80)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_GOLD = (255, 215, 0)
        
        # State constants
        self.STATE_EMPTY = 0
        self.STATE_PLANTED = 1
        self.STATE_GROWING = 2
        self.STATE_READY = 3

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # --- Game State Variables ---
        self.grid = []
        self.cursor_pos = [0, 0]
        self.harvested_crops = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.particles = []
        self.popups = []

        self.reset()
        
        # self.validate_implementation() # Optional: run validation on init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.harvested_crops = 0
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        
        self.grid = [[{'state': self.STATE_EMPTY, 'timer': 0} for _ in range(self.GRID_SIZE[1])] for _ in range(self.GRID_SIZE[0])]
        
        self.particles.clear()
        self.popups.clear()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Actions ---
        self._move_cursor(movement)
        
        if space_held:
            reward += self._action_plant_harvest()
            
        if shift_held:
            reward += self._action_sell()

        # --- Update Game State ---
        self._update_crops()
        self._update_particles()
        self._update_popups()
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.score >= self.WIN_COINS:
            reward += 100  # Goal-oriented reward
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward += -10 # Timeout penalty
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] -= 1
        elif movement == 2: # Down
            self.cursor_pos[1] += 1
        elif movement == 3: # Left
            self.cursor_pos[0] -= 1
        elif movement == 4: # Right
            self.cursor_pos[0] += 1
        
        # Clamp cursor position to grid
        self.cursor_pos[0] = max(0, min(self.GRID_SIZE[0] - 1, self.cursor_pos[0]))
        self.cursor_pos[1] = max(0, min(self.GRID_SIZE[1] - 1, self.cursor_pos[1]))

    def _action_plant_harvest(self):
        plot = self.grid[self.cursor_pos[0]][self.cursor_pos[1]]
        
        if plot['state'] == self.STATE_EMPTY:
            plot['state'] = self.STATE_PLANTED
            plot['timer'] = 0
            # Sound placeholder: pygame.mixer.Sound("plant.wav").play()
            return 0
            
        if plot['state'] == self.STATE_READY:
            plot['state'] = self.STATE_EMPTY
            plot['timer'] = 0
            self.harvested_crops += 1
            
            # Visual feedback for harvesting
            px, py = self._get_plot_screen_pos(self.cursor_pos[0], self.cursor_pos[1])
            self._create_popup("+1", (px + self.PLOT_SIZE[0] // 2, py))
            
            # Sound placeholder: pygame.mixer.Sound("harvest.wav").play()
            return 0.1 # Continuous feedback reward
        
        return 0

    def _action_sell(self):
        if self.harvested_crops > 0:
            coins_earned = self.harvested_crops * self.CROP_VALUE
            reward = self.harvested_crops * 1.0 # Event-based reward
            
            self.score += coins_earned
            
            # Visual feedback for selling
            for _ in range(min(30, self.harvested_crops)):
                self._create_coin_particle()

            self.harvested_crops = 0
            # Sound placeholder: pygame.mixer.Sound("cash_register.wav").play()
            return reward
        
        return 0
        
    def _update_crops(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                plot = self.grid[x][y]
                if plot['state'] in [self.STATE_PLANTED, self.STATE_GROWING]:
                    plot['timer'] += 1
                    if plot['timer'] >= self.GROWTH_TIME:
                        plot['state'] = self.STATE_READY
                        # Sound placeholder: pygame.mixer.Sound("ready.wav").play()
                    elif plot['timer'] > 0:
                         plot['state'] = self.STATE_GROWING

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid and crops
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                plot = self.grid[x][y]
                px, py = self._get_plot_screen_pos(x, y)
                rect = pygame.Rect(px, py, self.PLOT_SIZE[0], self.PLOT_SIZE[1])
                
                # Draw plot background
                pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, rect)
                
                if plot['state'] == self.STATE_PLANTED:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_PLANTED, rect, border_radius=3)
                    pygame.draw.circle(self.screen, self.COLOR_CROP_START, rect.center, 3)
                elif plot['state'] == self.STATE_GROWING:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_PLANTED, rect, border_radius=3)
                    progress = plot['timer'] / self.GROWTH_TIME
                    
                    # Interpolate color and size for growth animation
                    color = self._lerp_color(self.COLOR_CROP_START, self.COLOR_CROP_END, progress)
                    size = int(2 + (self.PLOT_SIZE[0] * 0.4) * progress)
                    pygame.draw.circle(self.screen, color, rect.center, size)
                elif plot['state'] == self.STATE_READY:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_PLANTED, rect, border_radius=3)
                    # Pulsing effect for ready crops
                    pulse = (math.sin(self.steps * 0.2) + 1) / 2
                    size = int(self.PLOT_SIZE[0] * 0.4 + pulse * 3)
                    pygame.draw.circle(self.screen, self.COLOR_CROP_READY, rect.center, size)
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, size, self.COLOR_GOLD)
        
        # Render particles and popups
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), p['size'])
        for pop in self.popups:
            text_surf = self.font_small.render(pop['text'], True, pop['color'])
            text_rect = text_surf.get_rect(center=pop['pos'])
            self.screen.blit(text_surf, text_rect)

        # Render cursor
        cursor_px, cursor_py = self._get_plot_screen_pos(self.cursor_pos[0], self.cursor_pos[1])
        cursor_rect = pygame.Rect(cursor_px, cursor_py, self.PLOT_SIZE[0], self.PLOT_SIZE[1])
        
        # Breathing effect for cursor border
        border_width = 2 + int((math.sin(self.steps * 0.15) + 1))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, border_width, border_radius=4)
        
    def _render_ui(self):
        # --- Score Display ---
        score_text = f"COINS: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_GOLD)
        self.screen.blit(score_surf, (20, 15))
        
        # --- Time Display ---
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {max(0, int(time_left))}"
        time_surf = self.font_large.render(time_text, True, self.COLOR_UI_TEXT)
        time_rect = time_surf.get_rect(topright=(self.WIDTH - 20, 15))
        self.screen.blit(time_surf, time_rect)
        
        # --- Harvested Crops Display ---
        harvest_text = f"HARVESTED: {self.harvested_crops}"
        harvest_surf = self.font_medium.render(harvest_text, True, self.COLOR_CROP_READY)
        harvest_rect = harvest_surf.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10))
        self.screen.blit(harvest_surf, harvest_rect)

        # --- Game Over Screen ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_COINS:
                end_text = "YOU WIN!"
                color = self.COLOR_GOLD
            else:
                end_text = "TIME'S UP!"
                color = (200, 50, 50)
            
            end_surf = self.font_large.render(end_text, True, color)
            end_rect = end_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_surf, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "harvested_crops": self.harvested_crops,
            "cursor_pos": self.cursor_pos,
        }
        
    def _get_plot_screen_pos(self, x, y):
        return (
            self.GRID_ORIGIN[0] + x * self.PLOT_SIZE[0],
            self.GRID_ORIGIN[1] + y * self.PLOT_SIZE[1]
        )

    def _lerp_color(self, c1, c2, t):
        t = max(0, min(1, t))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t)
        )

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.2 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
                
    def _create_coin_particle(self):
        start_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        target_pos = [100, 30]
        
        angle = self.np_random.uniform(-math.pi / 2, math.pi / 2)
        speed = self.np_random.uniform(4, 8)
        
        particle = {
            'pos': list(start_pos),
            'vel': [self.np_random.uniform(-5, 5), self.np_random.uniform(-8, -4)],
            'lifespan': 60,
            'color': self.COLOR_GOLD,
            'size': self.np_random.integers(3, 6)
        }
        self.particles.append(particle)

    def _update_popups(self):
        for pop in self.popups[:]:
            pop['pos'][1] -= 0.5 # Move up
            pop['lifespan'] -= 1
            alpha = max(0, min(255, int(255 * (pop['lifespan'] / pop['max_lifespan']))))
            pop['color'] = (pop['base_color'][0], pop['base_color'][1], pop['base_color'][2], alpha)
            if pop['lifespan'] <= 0:
                self.popups.remove(pop)
    
    def _create_popup(self, text, pos):
        popup = {
            'text': text,
            'pos': list(pos),
            'lifespan': 30,
            'max_lifespan': 30,
            'base_color': (255, 255, 255),
            'color': (255, 255, 255, 255)
        }
        self.popups.append(popup)
        
    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")