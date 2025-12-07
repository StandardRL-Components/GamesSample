
# Generated: 2025-08-28T03:40:54.470414
# Source Brief: brief_02075.md
# Brief Index: 2075

        
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
        "Controls: Use arrow keys to move the selector. Press space to plant a seed on empty soil or harvest a grown crop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small isometric farm. Plant seeds, wait for them to grow, and harvest them to earn coins. Reach 1000 coins before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.WIN_SCORE = 1000
        self.MAX_STEPS = 6000
        self.CROP_GROW_TIME = 200 # steps for a crop to mature
        self.HARVEST_REWARD_VALUE = 20 # coins per crop

        # State constants
        self.STATE_EMPTY = 0
        self.STATE_PLANTED = 1
        self.STATE_GROWN = 2

        # Visual constants
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = 24
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (25, 45, 40)
        self.COLOR_SOIL = (101, 67, 33)
        self.COLOR_SOIL_DARK = (71, 47, 23)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_PLANTED = (76, 175, 80)
        self.COLOR_GROWN = (255, 235, 59)
        self.COLOR_UI_TEXT = (255, 255, 240)
        self.COLOR_COIN = (255, 215, 0)
        
        # EXACT spaces:
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
        self.font_small = pygame.font.Font(None, 32)
        
        # Initialize state variables
        self.grid = None
        self.growth_timers = None
        self.selector_pos = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.particles = None
        
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), self.STATE_EMPTY, dtype=np.int8)
        self.growth_timers = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int16)
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.particles = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        
        reward = 0
        
        # 1. Handle player input
        self._handle_movement(movement)
        if space_held:
            reward += self._handle_action()

        # 2. Update game logic
        self._update_crops()
        self._update_particles()
        
        self.steps += 1
        
        # 3. Check for termination
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.selector_pos[1] = (self.selector_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: # Down
            self.selector_pos[1] = (self.selector_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            self.selector_pos[0] = (self.selector_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: # Right
            self.selector_pos[0] = (self.selector_pos[0] + 1) % self.GRID_SIZE

    def _handle_action(self):
        x, y = self.selector_pos
        current_state = self.grid[x, y]
        reward = 0

        if current_state == self.STATE_EMPTY:
            # Plant a seed
            self.grid[x, y] = self.STATE_PLANTED
            self.growth_timers[x, y] = 0
            # sfx: plant_seed.wav
        elif current_state == self.STATE_GROWN:
            # Harvest a crop
            prev_score_tier = self.score // 10
            self.score += self.HARVEST_REWARD_VALUE
            new_score_tier = self.score // 10
            
            self.grid[x, y] = self.STATE_EMPTY
            self.growth_timers[x, y] = 0
            
            reward += 0.1 # Continuous feedback
            if new_score_tier > prev_score_tier:
                reward += 1.0 * (new_score_tier - prev_score_tier) # Event-based reward

            # Spawn particles
            screen_pos = self._iso_to_screen(x, y)
            for _ in range(10):
                self._spawn_particle(screen_pos)
            # sfx: harvest_coin.wav
            
        return reward

    def _update_crops(self):
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                if self.grid[x, y] == self.STATE_PLANTED:
                    self.growth_timers[x, y] += 1
                    if self.growth_timers[x, y] >= self.CROP_GROW_TIME:
                        self.grid[x, y] = self.STATE_GROWN
                        # sfx: crop_ready.wav

    def _spawn_particle(self, pos):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 5)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed - 2]
        lifespan = random.randint(20, 40)
        self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifespan, 'max_life': lifespan})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT / 2)
        return int(screen_x), int(screen_y)

    def _draw_iso_tile(self, surface, grid_x, grid_y, color, border_color):
        points = [
            self._iso_to_screen(grid_x, grid_y + 1),
            self._iso_to_screen(grid_x + 1, grid_y + 1),
            self._iso_to_screen(grid_x + 1, grid_y),
            self._iso_to_screen(grid_x, grid_y),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, border_color)

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and crops
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                self._draw_iso_tile(self.screen, x, y, self.COLOR_SOIL, self.COLOR_SOIL_DARK)
                
                state = self.grid[x, y]
                if state == self.STATE_PLANTED or state == self.STATE_GROWN:
                    center_x, center_y = self._iso_to_screen(x + 0.5, y + 0.5)
                    
                    if state == self.STATE_PLANTED:
                        growth_ratio = min(1.0, self.growth_timers[x, y] / self.CROP_GROW_TIME)
                        radius = int(2 + 8 * growth_ratio)
                        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y - 8, radius, self.COLOR_PLANTED)
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y - 8, radius, self.COLOR_PLANTED)
                    elif state == self.STATE_GROWN:
                        radius = 12
                        # Pulsing effect for ready crops
                        pulse = abs(math.sin(self.steps * 0.1)) * 4
                        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y - 12, int(radius + pulse), self.COLOR_GROWN)
                        pygame.gfxdraw.aacircle(self.screen, center_x, center_y - 12, int(radius + pulse), self.COLOR_GROWN)

        # Draw selector
        sel_x, sel_y = self.selector_pos
        points = [
            self._iso_to_screen(sel_x, sel_y + 1),
            self._iso_to_screen(sel_x + 1, sel_y + 1),
            self._iso_to_screen(sel_x + 1, sel_y),
            self._iso_to_screen(sel_x, sel_y),
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 3)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = self.COLOR_COIN + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, color)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        coin_icon = self.font_large.render("⛁", True, self.COLOR_COIN)
        self.screen.blit(coin_icon, (10, 10))
        self.screen.blit(score_text, (45, 10))

        # Timer display
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_large.render(f"{time_left / 100:.2f}s", True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.WIN_SCORE:
                msg = "You Win!"
                color = self.COLOR_GROWN
            else:
                msg = "Time's Up!"
                color = (200, 50, 50)
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, self.MAX_STEPS - self.steps),
            "selector_pos": self.selector_pos,
        }

    def close(self):
        pygame.font.quit()
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Game loop
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not terminated:
            # Action mapping from keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        screen_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        frame = np.transpose(obs, (1, 0, 2))
        pygame.surfarray.blit_array(screen_surface, frame)
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for playability

    env.close()