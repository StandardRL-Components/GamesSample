
# Generated: 2025-08-27T15:26:56.000184
# Source Brief: brief_00996.md
# Brief Index: 996

        
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
        "Controls: ↑↓←→ to move the selector. Press space to plant a seed. Press shift to harvest a ripe crop and sell all harvested goods."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small isometric farm to harvest crops and sell them for profit within a time limit. Reach 1000 coins to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
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
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_SIZE = (10, 10)
        self.MAX_STEPS = 3000
        self.WIN_SCORE = 1000
        self.CROP_GROWTH_TIME = 250
        self.CROP_VALUE = 10  # coins per crop
        
        # Visuals
        self.TILE_WIDTH = 40
        self.TILE_HEIGHT = 20
        self.ORIGIN_X = self.screen_width // 2
        self.ORIGIN_Y = 100

        # Colors
        self.COLOR_BG = (40, 45, 50)
        self.COLOR_SOIL = (94, 63, 43)
        self.COLOR_SOIL_EDGE = (74, 43, 23)
        self.COLOR_SELECTOR = (0, 255, 255)
        self.COLOR_RIPE = (255, 220, 0)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_GOLD = (255, 215, 0)
        self.COLOR_TIMER = (200, 200, 200)
        self.GROWTH_COLORS = [
            (34, 139, 34),  # ForestGreen
            (50, 205, 50),  # LimeGreen
            (124, 252, 0), # LawnGreen
            (173, 255, 47), # GreenYellow
        ]

        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_pop = pygame.font.Font(None, 22)
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.selector_pos = None
        self.harvested_crops = 0
        self.last_action = None
        self.particles = []
        self.floating_texts = []
        
        # Initialize state
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = np.zeros(self.GRID_SIZE, dtype=int)
        self.selector_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.harvested_crops = 0
        self.last_action = np.array([0, 0, 0])
        self.particles = []
        self.floating_texts = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            action_reward = self._handle_input(action)
            reward += action_reward
            self._update_crops()

        self._update_particles()
        self._update_floating_texts()
        
        self.steps += 1
        self.last_action = action
        
        terminated = self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100
                self._create_floating_text("YOU WIN!", (self.screen_width // 2, self.screen_height // 2 - 30), self.COLOR_GOLD, 90, 60)
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        reward = 0
        movement, space_press, shift_press = action[0], action[1], action[2]
        
        # --- Movement (triggered on new non-zero move command) ---
        last_movement = self.last_action[0]
        if movement != 0 and movement != last_movement:
            if movement == 1: self.selector_pos[1] -= 1 # Up
            elif movement == 2: self.selector_pos[1] += 1 # Down
            elif movement == 3: self.selector_pos[0] -= 1 # Left
            elif movement == 4: self.selector_pos[0] += 1 # Right
            
            # Wrap around grid
            self.selector_pos[0] %= self.GRID_SIZE[0]
            self.selector_pos[1] %= self.GRID_SIZE[1]
            # sfx: selector_move.wav

        # --- Plant (triggered on rising edge of space bar) ---
        is_space_rising_edge = space_press == 1 and self.last_action[1] == 0
        if is_space_rising_edge:
            gx, gy = self.selector_pos
            if self.grid[gx, gy] == 0: # Can only plant on empty tile
                self.grid[gx, gy] = 1 # Start growing
                # sfx: plant_seed.wav
                cx, cy = self._iso_to_cart(gx, gy)
                self._create_particles(20, (cx, cy), self.COLOR_SOIL, 2)

        # --- Harvest & Sell (triggered on rising edge of shift key) ---
        is_shift_rising_edge = shift_press == 1 and self.last_action[2] == 0
        if is_shift_rising_edge:
            # Harvest
            gx, gy = self.selector_pos
            if self.grid[gx, gy] >= self.CROP_GROWTH_TIME:
                self.grid[gx, gy] = 0
                self.harvested_crops += 1
                reward += 1.0 # Reward for harvesting
                # sfx: harvest.wav
                cx, cy = self._iso_to_cart(gx, gy)
                self._create_particles(30, (cx, cy), self.COLOR_RIPE, 3)
            
            # Sell
            if self.harvested_crops > 0:
                coins_earned = self.harvested_crops * self.CROP_VALUE
                self.score += coins_earned
                reward += 1.0 * self.harvested_crops # Scaled reward for selling
                # sfx: cash_register.wav
                self._create_floating_text(f"+{coins_earned}", (60, 50), self.COLOR_GOLD, 60)
                self.harvested_crops = 0
        
        return reward

    def _update_crops(self):
        for x in range(self.GRID_SIZE[0]):
            for y in range(self.GRID_SIZE[1]):
                if 0 < self.grid[x, y] < self.CROP_GROWTH_TIME:
                    self.grid[x, y] += 1
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_selector()
        self._render_particles()
        self._render_floating_texts()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "harvested_crops": self.harvested_crops,
            "selector_pos": tuple(self.selector_pos),
        }

    def _iso_to_cart(self, gx, gy):
        cx = self.ORIGIN_X + (gx - gy) * (self.TILE_WIDTH / 2)
        cy = self.ORIGIN_Y + (gx + gy) * (self.TILE_HEIGHT / 2)
        return int(cx), int(cy)

    def _render_grid(self):
        for gy in range(self.GRID_SIZE[1]):
            for gx in range(self.GRID_SIZE[0]):
                cx, cy = self._iso_to_cart(gx, gy)
                
                # Draw tile
                points = [
                    (cx, cy - self.TILE_HEIGHT / 2),
                    (cx + self.TILE_WIDTH / 2, cy),
                    (cx, cy + self.TILE_HEIGHT / 2),
                    (cx - self.TILE_WIDTH / 2, cy),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_SOIL)
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_SOIL_EDGE)

                # Draw crop
                growth_stage = self.grid[gx, gy]
                if growth_stage > 0:
                    if growth_stage >= self.CROP_GROWTH_TIME:
                        # Ripe crop
                        pygame.draw.circle(self.screen, self.COLOR_RIPE, (cx, cy - 10), 8)
                        pygame.draw.circle(self.screen, (255,255,100), (cx, cy - 10), 8, 1)
                    else:
                        # Growing crop
                        progress = growth_stage / self.CROP_GROWTH_TIME
                        size = int(2 + 6 * progress)
                        color_index = min(len(self.GROWTH_COLORS) - 1, int(progress * len(self.GROWTH_COLORS)))
                        color = self.GROWTH_COLORS[color_index]
                        pygame.draw.circle(self.screen, color, (cx, cy - 10), size)

    def _render_selector(self):
        gx, gy = self.selector_pos
        cx, cy = self._iso_to_cart(gx, gy)
        points = [
            (cx, cy - self.TILE_HEIGHT / 2),
            (cx + self.TILE_WIDTH / 2, cy),
            (cx, cy + self.TILE_HEIGHT / 2),
            (cx - self.TILE_WIDTH / 2, cy),
        ]
        
        # Draw a thicker, glowing line for the selector
        for i in range(4):
            start = points[i]
            end = points[(i + 1) % 4]
            pygame.draw.aaline(self.screen, self.COLOR_SELECTOR, start, end, True)
            # Create a "glow" by drawing slightly offset lines
            pygame.draw.aaline(self.screen, (*self.COLOR_SELECTOR, 50), (start[0]+1, start[1]), (end[0]+1, end[1]), True)
            pygame.draw.aaline(self.screen, (*self.COLOR_SELECTOR, 50), (start[0]-1, start[1]), (end[0]-1, end[1]), True)
            pygame.draw.aaline(self.screen, (*self.COLOR_SELECTOR, 50), (start[0], start[1]+1), (end[0], end[1]+1), True)
            pygame.draw.aaline(self.screen, (*self.COLOR_SELECTOR, 50), (start[0], start[1]-1), (end[0], end[1]-1), True)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"{self.score}", True, self.COLOR_TEXT)
        pygame.draw.circle(self.screen, self.COLOR_GOLD, (25, 25), 10)
        pygame.draw.circle(self.screen, self.COLOR_TEXT, (25, 25), 10, 1)
        self.screen.blit(score_text, (45, 15))

        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_percent = time_left / self.MAX_STEPS
        timer_text = self.font_ui.render(f"{time_left / 50:.1f}s", True, self.COLOR_TEXT) # 50 steps/sec approx
        self.screen.blit(timer_text, (self.screen_width - 80, 15))

        # Timer bar
        bar_width = 150
        bar_height = 10
        bar_x = self.screen_width - bar_width - 20
        bar_y = 45
        pygame.draw.rect(self.screen, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIMER, (bar_x, bar_y, int(bar_width * time_percent), bar_height))

        # Harvested crops count (near bottom right)
        if self.harvested_crops > 0:
            harvest_text = self.font_ui.render(f"x{self.harvested_crops}", True, self.COLOR_TEXT)
            pygame.draw.circle(self.screen, self.COLOR_RIPE, (self.screen_width - 80, self.screen_height - 30), 10)
            self.screen.blit(harvest_text, (self.screen_width - 60, self.screen_height - 40))

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            if self.score < self.WIN_SCORE:
                end_text = pygame.font.Font(None, 60).render("TIME'S UP", True, self.COLOR_TIMER)
                text_rect = end_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 30))
                self.screen.blit(end_text, text_rect)

    def _create_particles(self, count, pos, color, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel_mag = random.uniform(0.5, speed)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * vel_mag, math.sin(angle) * vel_mag - 1.5], # Move up slightly
                'lifespan': random.randint(15, 30),
                'color': color,
                'size': random.randint(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30))
            alpha = max(0, min(255, alpha))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

    def _create_floating_text(self, text, pos, color, lifespan, size=22):
        font = pygame.font.Font(None, size)
        self.floating_texts.append({
            'text': text,
            'pos': list(pos),
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'color': color,
            'font': font
        })

    def _update_floating_texts(self):
        for ft in self.floating_texts:
            ft['pos'][1] -= 0.5
            ft['lifespan'] -= 1
        self.floating_texts = [ft for ft in self.floating_texts if ft['lifespan'] > 0]

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            alpha = int(255 * (ft['lifespan'] / ft['max_lifespan']))
            alpha = max(0, min(255, alpha))
            text_surf = ft['font'].render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            text_rect = text_surf.get_rect(center=(int(ft['pos'][0]), int(ft['pos'][1])))
            self.screen.blit(text_surf, text_rect)
            
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

# Example usage to run and visualize the game
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Isometric Farm")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        
        for key, move_val in key_to_action.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = np.array([movement_action, space_action, shift_action])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(50) # Run at 50 FPS

    env.close()