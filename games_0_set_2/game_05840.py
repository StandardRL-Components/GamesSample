
# Generated: 2025-08-28T06:14:52.885650
# Source Brief: brief_05840.md
# Brief Index: 5840

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrows to move cursor. Space to plant, harvest, or sell."
    )

    # User-facing game description
    game_description = (
        "Procedurally generated top-down farming simulator. Plant, harvest, and sell crops to earn 1000 coins before time runs out."
    )

    # Frames advance on action, not automatically
    auto_advance = False

    # --- Constants ---
    # Game parameters
    GRID_COLS, GRID_ROWS = 12, 7
    PLOT_SIZE = (50, 45)
    PLOT_GAP = 2
    CROP_GROWTH_TIME = 50
    CROP_SELL_VALUE = 10
    MAX_STEPS = 6000
    WIN_SCORE = 1000

    # Rewards
    REWARD_PLANT = 0.1
    REWARD_HARVEST = 0.2
    REWARD_SELL = 1.0
    REWARD_WIN = 100.0
    REWARD_LOSE = -100.0

    # Colors
    COLOR_BG = (34, 32, 52)
    COLOR_UI_BG = (24, 22, 42)
    COLOR_GRID_BG = (54, 52, 72)
    COLOR_PLOT_EMPTY = (94, 63, 45)
    COLOR_PLOT_PLANTED = (45, 94, 63)
    COLOR_PLOT_READY = (212, 175, 55)
    COLOR_SELL_POINT = (60, 100, 180)
    COLOR_SELL_POINT_TEXT = (220, 230, 255)
    COLOR_CURSOR = (255, 255, 255, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_SHADOW = (20, 20, 20)

    # Plot States
    STATE_EMPTY = 0
    STATE_PLANTED = 1
    STATE_READY = 2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.screen_width, self.screen_height = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_m = pygame.font.Font(None, 28)
        self.font_s = pygame.font.Font(None, 22)

        # Layout calculation
        self.grid_width = self.GRID_COLS * (self.PLOT_SIZE[0] + self.PLOT_GAP) - self.PLOT_GAP
        self.grid_height = self.GRID_ROWS * (self.PLOT_SIZE[1] + self.PLOT_GAP) - self.PLOT_GAP
        self.grid_offset_x = (self.screen_width - self.grid_width) // 2
        self.grid_offset_y = 60

        self.sell_point_rect = pygame.Rect(
            self.screen_width // 2 - 75,
            self.grid_offset_y + self.grid_height + 15,
            150, 35
        )
        
        # Initialize state variables
        self.farm_plots = []
        self.cursor_pos = [0, 0]
        self.harvested_crops = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.effects = []
        
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.harvested_crops = 0
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.effects = []

        self.farm_plots = [
            [{'state': self.STATE_EMPTY, 'growth': 0} for _ in range(self.GRID_ROWS)]
            for _ in range(self.GRID_COLS)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.0
        
        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- 1. Handle Player Input ---
        self._handle_movement(movement)
        
        if space_pressed:
            reward += self._handle_interaction()

        # --- 2. Update World State ---
        self._update_crops()
        self._update_effects()
        
        # --- 3. Update Timers & Check Termination ---
        self.steps += 1
        
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += self.REWARD_WIN
            terminated = True
            self.game_over = True
            self._add_effect(self.screen_width/2, self.screen_height/2, "VICTORY!", self.COLOR_PLOT_READY, 120)

        elif self.steps >= self.MAX_STEPS:
            reward += self.REWARD_LOSE
            terminated = True
            self.game_over = True
            self._add_effect(self.screen_width/2, self.screen_height/2, "TIME UP", self.COLOR_SELL_POINT, 120)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

    def _handle_interaction(self):
        cursor_screen_x = self.grid_offset_x + self.cursor_pos[0] * (self.PLOT_SIZE[0] + self.PLOT_GAP)
        cursor_screen_y = self.grid_offset_y + self.cursor_pos[1] * (self.PLOT_SIZE[1] + self.PLOT_GAP)
        cursor_rect = pygame.Rect(cursor_screen_x, cursor_screen_y, self.PLOT_SIZE[0], self.PLOT_SIZE[1])

        # Check for interaction with sell point
        if self.sell_point_rect.colliderect(cursor_rect):
            if self.harvested_crops > 0:
                # sfx: cash register sound
                sold_amount = self.harvested_crops * self.CROP_SELL_VALUE
                self.score += sold_amount
                self.harvested_crops = 0
                self._add_effect(self.sell_point_rect.centerx, self.sell_point_rect.centery, f"+{sold_amount} COINS", self.COLOR_PLOT_READY, 30)
                return self.REWARD_SELL
            return 0.0

        # Interaction with farm plot
        x, y = self.cursor_pos
        plot = self.farm_plots[x][y]

        if plot['state'] == self.STATE_EMPTY:
            # sfx: planting seed sound
            plot['state'] = self.STATE_PLANTED
            plot['growth'] = 0
            self._add_effect(cursor_rect.centerx, cursor_rect.centery, "PLANT", self.COLOR_PLOT_PLANTED, 20)
            return self.REWARD_PLANT
        
        elif plot['state'] == self.STATE_READY:
            # sfx: harvesting pop sound
            plot['state'] = self.STATE_EMPTY
            plot['growth'] = 0
            self.harvested_crops += 1
            self._add_effect(cursor_rect.centerx, cursor_rect.centery, "HARVEST", self.COLOR_PLOT_READY, 20)
            return self.REWARD_HARVEST
            
        return 0.0

    def _update_crops(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                plot = self.farm_plots[x][y]
                if plot['state'] == self.STATE_PLANTED:
                    plot['growth'] += 1
                    if plot['growth'] >= self.CROP_GROWTH_TIME:
                        plot['state'] = self.STATE_READY

    def _add_effect(self, x, y, text, color, lifetime):
        self.effects.append({'x': x, 'y': y, 'text': text, 'color': color, 'lifetime': lifetime, 'max_lifetime': lifetime})

    def _update_effects(self):
        self.effects = [e for e in self.effects if e['lifetime'] > 0]
        for effect in self.effects:
            effect['lifetime'] -= 1
            effect['y'] -= 0.5 # Effects float upwards

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_text(self, text, font, color, pos, shadow=True):
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (pos[0] + 1, pos[1] + 1))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(
            self.screen, self.COLOR_GRID_BG,
            (self.grid_offset_x, self.grid_offset_y, self.grid_width, self.grid_height),
            border_radius=5
        )

        # Draw plots
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                plot = self.farm_plots[x][y]
                plot_rect = pygame.Rect(
                    self.grid_offset_x + x * (self.PLOT_SIZE[0] + self.PLOT_GAP),
                    self.grid_offset_y + y * (self.PLOT_SIZE[1] + self.PLOT_GAP),
                    self.PLOT_SIZE[0], self.PLOT_SIZE[1]
                )
                
                if plot['state'] == self.STATE_EMPTY:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, plot_rect, border_radius=3)
                elif plot['state'] == self.STATE_PLANTED:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_PLANTED, plot_rect, border_radius=3)
                    # Draw growing crop
                    growth_ratio = plot['growth'] / self.CROP_GROWTH_TIME
                    radius = int(min(self.PLOT_SIZE) / 2 * growth_ratio)
                    if radius > 0:
                        pygame.gfxdraw.filled_circle(self.screen, plot_rect.centerx, plot_rect.centery, radius, (113, 168, 86))
                        pygame.gfxdraw.aacircle(self.screen, plot_rect.centerx, plot_rect.centery, radius, (113, 168, 86))
                elif plot['state'] == self.STATE_READY:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_READY, plot_rect, border_radius=3)

        # Draw sell point
        pygame.draw.rect(self.screen, self.COLOR_SELL_POINT, self.sell_point_rect, border_radius=5)
        sell_text = self.font_m.render("SELL", True, self.COLOR_SELL_POINT_TEXT)
        self.screen.blit(sell_text, sell_text.get_rect(center=self.sell_point_rect.center))

        # Draw cursor
        cursor_screen_x = self.grid_offset_x + self.cursor_pos[0] * (self.PLOT_SIZE[0] + self.PLOT_GAP)
        cursor_screen_y = self.grid_offset_y + self.cursor_pos[1] * (self.PLOT_SIZE[1] + self.PLOT_GAP)
        cursor_rect = pygame.Rect(cursor_screen_x, cursor_screen_y, self.PLOT_SIZE[0], self.PLOT_SIZE[1])
        
        # Check if cursor is over sell point
        if self.sell_point_rect.colliderect(cursor_rect):
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, self.sell_point_rect, 5, border_radius=5)
        else:
            pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=3)

        # Draw effects
        for effect in self.effects:
            alpha = int(255 * (effect['lifetime'] / effect['max_lifetime']))
            color = (*effect['color'], alpha)
            font = self.font_s if effect['text'] in ["PLANT", "HARVEST"] else self.font_m
            text_surf = font.render(effect['text'], True, effect['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(effect['x']), int(effect['y'])))
            self.screen.blit(text_surf, text_rect)
            
    def _render_ui(self):
        # UI Background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.screen_width, 50))
        pygame.draw.line(self.screen, self.COLOR_GRID_BG, (0, 50), (self.screen_width, 50), 2)

        # Score
        score_text = f"COINS: {self.score} / {self.WIN_SCORE}"
        self._render_text(score_text, self.font_m, self.COLOR_TEXT, (15, 15))

        # Harvested Crops
        harvest_text = f"INVENTORY: {self.harvested_crops}"
        harvest_text_surf = self.font_m.render(harvest_text, True, self.COLOR_TEXT)
        self._render_text(harvest_text, self.font_m, self.COLOR_TEXT, (self.screen_width // 2 - harvest_text_surf.get_width() // 2, 15))
        
        # Time
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = f"STEPS: {time_left}"
        time_surf = self.font_m.render(time_text, True, self.COLOR_TEXT)
        self._render_text(time_text, self.font_m, self.COLOR_TEXT, (self.screen_width - time_surf.get_width() - 15, 15))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "harvested_crops": self.harvested_crops,
            "cursor_pos": list(self.cursor_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)

        # Test brief-specific assertions
        self.reset()
        for _ in range(51):
            self.step(self.action_space.sample())
        assert -10 < self.score < 1000, f"Score after 51 random steps is {self.score}, expected between -10 and 1000"
        
        print("✓ Implementation validated successfully")

# Example usage for manual testing
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for visualization ---
    pygame.display.set_caption("Farming Simulator")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print("      FARMING SIMULATOR")
    print("="*30)
    print(env.game_description)
    print("\n" + env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Human Input ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4

        # Interaction
        if keys[pygame.K_SPACE]:
            action[1] = 1

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(15) # Limit frame rate for human playability

    print(f"Game Over! Final Score: {info['score']} in {info['steps']} steps.")
    env.close()