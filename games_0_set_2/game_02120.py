
# Generated: 2025-08-27T19:19:24.659507
# Source Brief: brief_02120.md
# Brief Index: 2120

        
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
        "Controls: ↑↓←→ to move the cursor. Press space to plant, harvest, or sell crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Cultivate crops in a grid-based farm to earn 500 gold before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (34, 52, 54)
    COLOR_GRID = (44, 62, 64)
    COLOR_PLOT_EMPTY = (94, 72, 54)
    COLOR_PLOT_PLANTED = (44, 82, 54)
    COLOR_CROP_GROWING = (104, 153, 72)
    COLOR_CROP_READY = (255, 225, 102)
    COLOR_MARKET = (133, 94, 66)
    COLOR_MARKET_ROOF = (181, 83, 83)
    COLOR_CURSOR = (230, 255, 255)
    COLOR_CURSOR_GLOW = (180, 220, 255, 50)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_TIMER_BAR_BG = (50, 50, 50)
    COLOR_TIMER_BAR_FG = (200, 200, 50)
    COLOR_GOLD = (255, 215, 0)
    
    # Game Parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 12
    GRID_ROWS = 6
    CELL_SIZE = 40
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_COLS * CELL_SIZE) // 2
    GRID_MARGIN_Y = 60
    
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    
    CROP_GROW_TIME_STEPS = 5 * FPS # 5 seconds to grow
    WIN_GOLD = 500
    
    # Plot States
    STATE_EMPTY = 0
    STATE_PLANTED = 1
    STATE_READY = 2
    
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.grid = None
        self.growth_timers = None
        self.cursor_pos = None
        self.player_gold = None
        self.player_inventory = None
        self.game_timer = None
        self.game_over = None
        self.win_condition_met = None
        self.steps = None
        self.space_was_held = None
        self.particles = None
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.full((self.GRID_COLS, self.GRID_ROWS), self.STATE_EMPTY, dtype=np.int8)
        self.growth_timers = np.zeros((self.GRID_COLS, self.GRID_ROWS), dtype=np.int32)
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.player_gold = 0
        self.player_inventory = 0
        self.game_timer = self.MAX_STEPS
        
        self.game_over = False
        self.win_condition_met = False
        self.steps = 0
        self.space_was_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Time and Automatic Processes ---
        self.steps += 1
        self.game_timer -= 1
        self._update_crop_growth()
        self._update_particles()
        
        # --- Handle Player Actions ---
        movement = action[0]
        space_held = action[1] == 1
        space_pressed = space_held and not self.space_was_held
        
        # 1. Move Cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS
        
        # 2. Handle 'Space' Action (Plant/Harvest/Sell)
        if space_pressed:
            cx, cy = self.cursor_pos
            is_market_row = (cy == self.GRID_ROWS - 1)
            
            # Sell at market
            if is_market_row and self.player_inventory > 0:
                # SFX: Cha-ching!
                sold_count = self.player_inventory
                gold_earned = sold_count * 10
                self.player_gold += gold_earned
                self.player_inventory = 0
                reward += 1.0 * sold_count
                self._create_gold_particles(sold_count)
            # Harvest
            elif self.grid[cx, cy] == self.STATE_READY:
                # SFX: Harvest rustle
                self.grid[cx, cy] = self.STATE_EMPTY
                self.player_inventory += 1
                reward += 0.2
            # Plant
            elif self.grid[cx, cy] == self.STATE_EMPTY:
                # SFX: Plant seed
                self.grid[cx, cy] = self.STATE_PLANTED
                self.growth_timers[cx, cy] = 0
                reward += 0.1
                
        self.space_was_held = space_held
        
        # --- Check for Termination ---
        if not self.game_over:
            if self.player_gold >= self.WIN_GOLD:
                self.win_condition_met = True
                self.game_over = True
                reward += 100
            elif self.game_timer <= 0:
                self.game_over = True
                reward -= 100
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _update_crop_growth(self):
        for x in range(self.GRID_COLS):
            for y in range(self.GRID_ROWS):
                if self.grid[x, y] == self.STATE_PLANTED:
                    self.growth_timers[x, y] += 1
                    if self.growth_timers[x, y] >= self.CROP_GROW_TIME_STEPS:
                        self.grid[x, y] = self.STATE_READY
                        # SFX: Ding! Crop ready
                        
    def _create_gold_particles(self, count):
        market_rect = pygame.Rect(self.GRID_MARGIN_X, self.SCREEN_HEIGHT - 50, self.GRID_COLS * self.CELL_SIZE, 50)
        for _ in range(count * 3):
            particle = {
                "pos": [random.uniform(market_rect.left, market_rect.right), market_rect.top + 10],
                "vel": [random.uniform(-1, 1), random.uniform(-3, -1)],
                "life": random.randint(20, 40),
                "color": self.COLOR_GOLD,
                "radius": random.randint(2, 4)
            }
            self.particles.append(particle)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.player_gold,
            "steps": self.steps,
            "inventory": self.player_inventory,
            "time_left": self.game_timer
        }

    def _render_game(self):
        self._render_grid()
        self._render_market()
        self._render_cursor()
        self._render_particles()

    def _render_grid(self):
        for y in range(self.GRID_ROWS - 1): # Exclude market row
            for x in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_MARGIN_X + x * self.CELL_SIZE,
                    self.GRID_MARGIN_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                # Draw plot background
                pygame.draw.rect(self.screen, self.COLOR_PLOT_EMPTY, rect)
                
                state = self.grid[x, y]
                if state == self.STATE_PLANTED:
                    pygame.draw.rect(self.screen, self.COLOR_PLOT_PLANTED, rect.inflate(-4, -4))
                    growth_progress = self.growth_timers[x, y] / self.CROP_GROW_TIME_STEPS
                    crop_size = int(self.CELL_SIZE * 0.6 * growth_progress)
                    crop_rect = pygame.Rect(0, 0, crop_size, crop_size)
                    crop_rect.center = rect.center
                    pygame.draw.rect(self.screen, self.COLOR_CROP_GROWING, crop_rect, border_radius=crop_size//2)
                elif state == self.STATE_READY:
                    pygame.draw.rect(self.screen, self.COLOR_CROP_READY, rect.inflate(-4, -4), border_radius=8)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_market(self):
        market_rect = pygame.Rect(
            self.GRID_MARGIN_X,
            self.GRID_MARGIN_Y + (self.GRID_ROWS - 1) * self.CELL_SIZE,
            self.GRID_COLS * self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_MARKET, market_rect)
        pygame.draw.rect(self.screen, self.COLOR_MARKET_ROOF, market_rect, 6)
        
        market_text = self.font_small.render("MARKET (SELL HERE)", True, self.COLOR_UI_TEXT)
        text_rect = market_text.get_rect(center=market_rect.center)
        self.screen.blit(market_text, text_rect)
        
    def _render_cursor(self):
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + cx * self.CELL_SIZE,
            self.GRID_MARGIN_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        
        # Glow effect
        glow_surf = pygame.Surface((self.CELL_SIZE * 1.5, self.CELL_SIZE * 1.5), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_CURSOR_GLOW, glow_surf.get_rect(), border_radius=12)
        self.screen.blit(glow_surf, glow_surf.get_rect(center=cursor_rect.center))
        
        # Main cursor
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

    def _render_particles(self):
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), p['radius'])

    def _render_ui(self):
        # Timer Bar
        timer_width = self.SCREEN_WIDTH - 20
        timer_height = 20
        timer_x = 10
        timer_y = 10
        
        progress = max(0, self.game_timer / self.MAX_STEPS)
        
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (timer_x, timer_y, timer_width, timer_height), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_FG, (timer_x, timer_y, int(timer_width * progress), timer_height), border_radius=5)
        
        # Gold and Inventory Display
        ui_panel_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 40, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, (0,0,0,150), ui_panel_rect)
        
        gold_text = self.font_large.render(f"GOLD: {self.player_gold} / {self.WIN_GOLD}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (20, self.SCREEN_HEIGHT - 32))
        
        inventory_text = self.font_large.render(f"CROPS: {self.player_inventory}", True, self.COLOR_CROP_GROWING)
        inv_rect = inventory_text.get_rect(right=self.SCREEN_WIDTH - 20, top=self.SCREEN_HEIGHT - 32)
        self.screen.blit(inventory_text, inv_rect)
        
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win_condition_met:
                msg = "YOU WIN!"
                color = self.COLOR_GOLD
            else:
                msg = "TIME'S UP!"
                color = self.COLOR_MARKET_ROOF
                
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Farming Game")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    
    # --- Action state ---
    movement = 0 # 0=none, 1=up, 2=down, 3=left, 4=right
    space_held = 0 # 0=released, 1=held
    shift_held = 0 # 0=released, 1=held

    print(GameEnv.user_guide)
    print(GameEnv.game_description)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Key Down
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4
                elif event.key == pygame.K_SPACE: space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 1
                elif event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                    terminated = False
            
            # Key Up
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    movement = 0
                elif event.key == pygame.K_SPACE: space_held = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_held = 0

        if not terminated:
            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)

        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()