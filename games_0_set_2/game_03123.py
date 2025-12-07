
# Generated: 2025-08-28T07:03:23.012203
# Source Brief: brief_03123.md
# Brief Index: 3123

        
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
        "Controls: Arrow keys to move selector. Space to plant/harvest. Shift to sell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small farm to earn $1000. Plant seeds, wait for them to grow, harvest the crops, and sell them at the barn. Beat the 5-minute timer!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.MAX_STEPS = 5000
        self.WIN_PROFIT = 1000.0
        self.CROP_GROW_TIME = 300  # in steps, ~10s at 30fps
        self.CROP_PRICE = 5.0

        # Plot States
        self.STATE_EMPTY = 0
        self.STATE_GROWING = 1
        self.STATE_READY = 2
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_ui = pygame.font.Font(None, 28)
        self.font_end = pygame.font.Font(None, 72)
        self.font_subtitle = pygame.font.Font(None, 36)

        # Colors
        self.COLOR_BG = (20, 30, 25)
        self.COLOR_GRID_LINE = (74, 43, 23)
        self.COLOR_PLOT_EMPTY = (94, 63, 43)
        self.COLOR_PLOT_GROWING = (54, 83, 23)
        self.COLOR_PLOT_READY = (134, 103, 33)
        self.COLOR_CROP_GROWING = (76, 175, 80)
        self.COLOR_CROP_READY = (255, 235, 59)
        self.COLOR_SELECTOR = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (20, 20, 20)
        self.COLOR_TIMER_BAR = (63, 81, 181)
        self.COLOR_TIMER_BG = (40, 51, 114)
        self.COLOR_BARN = (121, 85, 72)
        self.COLOR_BARN_ROOF = (183, 28, 28)
        self.COLOR_HARVEST_PARTICLE = (255, 235, 59)
        self.COLOR_SELL_PARTICLE = (139, 195, 74)

        # Isometric projection helpers
        self.TILE_WIDTH = 40
        self.TILE_HEIGHT = 20
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 120

        # State variables (initialized in reset)
        self.steps = None
        self.profit = None
        self.game_over = None
        self.win = None
        self.harvested_crops = None
        self.selector_pos = None
        self.farm_grid = None
        self.growth_timers = None
        self.last_space_held = None
        self.last_shift_held = None
        self.particles = None
        
        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.profit = 0.0
        self.game_over = False
        self.win = False
        self.harvested_crops = 0
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.farm_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.growth_timers = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return (
                self._get_observation(),
                0,
                True,
                False,
                self._get_info()
            )

        self.steps += 1

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Detect single presses
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # Update crop growth
        growing_plots = self.farm_grid == self.STATE_GROWING
        self.growth_timers[growing_plots] += 1
        ready_plots = (self.growth_timers >= self.CROP_GROW_TIME) & growing_plots
        self.farm_grid[ready_plots] = self.STATE_READY

        # Handle player movement
        if movement == 1: self.selector_pos[0] -= 1  # Up
        elif movement == 2: self.selector_pos[0] += 1  # Down
        elif movement == 3: self.selector_pos[1] -= 1  # Left
        elif movement == 4: self.selector_pos[1] += 1  # Right
        
        # Wrap selector around grid
        self.selector_pos[0] %= self.GRID_SIZE
        self.selector_pos[1] %= self.GRID_SIZE
        
        sel_x, sel_y = self.selector_pos

        # Handle plant/harvest action
        if space_press:
            plot_state = self.farm_grid[sel_x, sel_y]
            if plot_state == self.STATE_EMPTY:
                self.farm_grid[sel_x, sel_y] = self.STATE_GROWING
                self.growth_timers[sel_x, sel_y] = 0
                # Sound: plant seed
            elif plot_state == self.STATE_READY:
                self.farm_grid[sel_x, sel_y] = self.STATE_EMPTY
                self.harvested_crops += 1
                reward += 0.1
                self._spawn_particles(sel_x, sel_y, 'harvest')
                # Sound: harvest crop

        # Handle sell action
        if shift_press and self.harvested_crops > 0:
            earnings = self.harvested_crops * self.CROP_PRICE
            self.profit += earnings
            self.harvested_crops = 0
            reward += 1.0
            self._spawn_particles(0, 0, 'sell')
            # Sound: cash register
        
        # No-op penalty
        if movement == 0 and not space_held and not shift_held:
            reward -= 0.01

        # Update particles
        self._update_particles()

        # Check termination conditions
        if self.profit >= self.WIN_PROFIT:
            self.profit = self.WIN_PROFIT # cap profit
            reward += 100
            terminated = True
            self.game_over = True
            self.win = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            self.win = False

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_farm()
        self._render_barn()
        self._render_particles()
        self._render_ui()
        
        if self.game_over:
            self._render_end_screen()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.profit,
            "steps": self.steps,
            "harvested_crops": self.harvested_crops,
        }

    def _grid_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _render_farm(self):
        # Render from back to front for correct layering
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                center_x, center_y = self._grid_to_screen(r, c)
                
                # Determine plot color
                plot_state = self.farm_grid[r, c]
                if plot_state == self.STATE_EMPTY:
                    plot_color = self.COLOR_PLOT_EMPTY
                elif plot_state == self.STATE_GROWING:
                    plot_color = self.COLOR_PLOT_GROWING
                else: # STATE_READY
                    plot_color = self.COLOR_PLOT_READY
                
                # Draw isometric tile
                points = [
                    (center_x, center_y - self.TILE_HEIGHT / 2),
                    (center_x + self.TILE_WIDTH / 2, center_y),
                    (center_x, center_y + self.TILE_HEIGHT / 2),
                    (center_x - self.TILE_WIDTH / 2, center_y)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, plot_color)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID_LINE)

                # Draw crop
                if plot_state == self.STATE_GROWING:
                    growth_ratio = min(1.0, self.growth_timers[r, c] / self.CROP_GROW_TIME)
                    radius = int(2 + 5 * growth_ratio)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y - 5, radius, self.COLOR_CROP_GROWING)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y - 5, radius, self.COLOR_CROP_GROWING)
                elif plot_state == self.STATE_READY:
                    radius = 8
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y - 5, radius, self.COLOR_CROP_READY)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y - 5, radius, self.COLOR_CROP_READY)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        center_x, center_y = self._grid_to_screen(sel_x, sel_y)
        points = [
            (center_x, center_y - self.TILE_HEIGHT / 2),
            (center_x + self.TILE_WIDTH / 2, center_y),
            (center_x, center_y + self.TILE_HEIGHT / 2),
            (center_x - self.TILE_WIDTH / 2, center_y)
        ]
        # Pulsating alpha for selector
        alpha = int(128 + 127 * math.sin(self.steps * 0.2))
        selector_color = self.COLOR_SELECTOR + (alpha,)
        
        # Draw a thicker line for the selector
        pygame.draw.lines(self.screen, selector_color, True, points, 3)

    def _render_barn(self):
        barn_rect = pygame.Rect(20, 150, 80, 80)
        roof_points = [(15, 150), (60, 110), (105, 150)]
        pygame.draw.rect(self.screen, self.COLOR_BARN, barn_rect)
        pygame.draw.polygon(self.screen, self.COLOR_BARN_ROOF, roof_points)
        
        # Light up if crops are stored
        if self.harvested_crops > 0:
            pygame.gfxdraw.filled_circle(self.screen, 60, 140, 5, self.COLOR_CROP_READY)
            pygame.gfxdraw.aacircle(self.screen, 60, 140, 5, self.COLOR_CROP_READY)

    def _render_ui(self):
        # Timer Bar
        timer_width = self.WIDTH - 20
        progress = self.steps / self.MAX_STEPS
        bar_fill = int(timer_width * (1 - progress))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BG, (10, 10, timer_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (10, 10, bar_fill, 20))

        # Profit Text
        profit_text = f"Profit: ${self.profit:,.2f}"
        self._draw_text(profit_text, self.font_ui, self.COLOR_TEXT, (15, 40), self.COLOR_TEXT_SHADOW)
        
        # Storage Text
        storage_text = f"Storage: {self.harvested_crops}"
        self._draw_text(storage_text, self.font_ui, self.COLOR_TEXT, (15, 70), self.COLOR_TEXT_SHADOW)

    def _render_end_screen(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if self.win:
            title_text = "You Win!"
            color = (150, 255, 150)
            subtitle_text = f"Final Profit: ${self.profit:,.2f}"
        else:
            title_text = "Time's Up!"
            color = (255, 150, 150)
            subtitle_text = "Better luck next time!"
            
        self._draw_text(title_text, self.font_end, color, (self.WIDTH / 2, self.HEIGHT / 2 - 20), self.COLOR_TEXT_SHADOW, center=True)
        self._draw_text(subtitle_text, self.font_subtitle, self.COLOR_TEXT, (self.WIDTH / 2, self.HEIGHT / 2 + 30), self.COLOR_TEXT_SHADOW, center=True)

    def _draw_text(self, text, font, color, pos, shadow_color=None, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center: text_rect.center = pos
        else: text_rect.topleft = pos

        if shadow_color:
            shadow_surface = font.render(text, True, shadow_color)
            shadow_rect = shadow_surface.get_rect()
            if center: shadow_rect.center = (pos[0] + 2, pos[1] + 2)
            else: shadow_rect.topleft = (pos[0] + 2, pos[1] + 2)
            self.screen.blit(shadow_surface, shadow_rect)
        
        self.screen.blit(text_surface, text_rect)

    def _spawn_particles(self, grid_x, grid_y, p_type):
        if p_type == 'harvest':
            count = 10
            color = self.COLOR_HARVEST_PARTICLE
            sx, sy = self._grid_to_screen(grid_x, grid_y)
            for _ in range(count):
                self.particles.append({
                    'pos': [sx, sy - 5],
                    'vel': [random.uniform(-1, 1), random.uniform(-3, -1)],
                    'life': random.randint(20, 40),
                    'color': color
                })
        elif p_type == 'sell':
            count = 20
            color = self.COLOR_SELL_PARTICLE
            for _ in range(count):
                 self.particles.append({
                    'pos': [random.randint(15, 150), 40],
                    'vel': [random.uniform(-0.5, 0.5), random.uniform(-2, -0.5)],
                    'life': random.randint(30, 50),
                    'color': color
                })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 50))
            alpha = max(0, min(255, alpha))
            color = p['color'] + (alpha,)
            size = int(max(1, 5 * (p['life'] / 50)))
            pygame.draw.circle(self.screen, color, p['pos'], size)

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

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    # env.validate_implementation() # Run validation
    
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Farm Manager")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        space = 0 # released
        shift = 0 # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']}. Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(30) # Limit to 30 FPS

    env.close()