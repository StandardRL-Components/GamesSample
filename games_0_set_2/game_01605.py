
# Generated: 2025-08-28T02:06:39.652771
# Source Brief: brief_01605.md
# Brief Index: 1605

        
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


# --- Helper Particle Class for Visual Effects ---
class Particle:
    def __init__(self, x, y, color, size, life, dx, dy):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.dx = dx
        self.dy = dy
        self.gravity = 0.1

    def update(self):
        self.life -= 1
        self.x += self.dx
        self.y += self.dy
        self.dy += self.gravity
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            temp_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
            surface.blit(temp_surf, (self.x - self.size, self.y - self.size), special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to plant or harvest. Press Shift to sell harvested crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Manage a small farm. Plant seeds, harvest ripe crops, and sell them to reach the profit goal before time runs out."
    )

    # Frames auto-advance at 30fps for smooth growth and timers.
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 10
    WIN_SCORE = 1000
    FPS = 30
    GAME_DURATION_SECONDS = 120
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Game Mechanics
    INITIAL_COINS = 50
    PLANT_COST = 5
    CROP_SELL_PRICE = 20
    GROWTH_STAGE_TIME = 5 * FPS # 5 seconds per stage

    # Plot States
    STATE_EMPTY = 0
    STATE_SEED = 1
    STATE_GROWING = 2
    STATE_RIPE = 3

    # Colors
    COLOR_BG = (25, 42, 28)
    COLOR_SOIL = (89, 61, 43)
    COLOR_GRID = (69, 41, 23)
    COLOR_SEED = (144, 238, 144)
    COLOR_GROWING = (34, 139, 34)
    COLOR_RIPE = (255, 215, 0)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_UI_TEXT = (255, 255, 240)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_TIME_BAR = (70, 130, 180)
    COLOR_TIME_BAR_WARN = (255, 69, 0)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        self.grid_width = self.GRID_SIZE * 32
        self.grid_height = self.GRID_SIZE * 32
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = self.INITIAL_COINS
        self.game_over = False
        self.win = False

        self.cursor_pos = np.array([self.GRID_SIZE // 2, self.GRID_SIZE // 2])
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.growth_timers = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.harvested_crops = 0
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Process Actions
        reward += self._handle_actions(action)

        # 2. Update Game State (Crop Growth)
        self._update_crops()

        # 3. Update Particles
        self._update_particles()
        
        # 4. Update Timers and Check Termination
        self.steps += 1
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win:
                reward += 100  # Win reward
            else:
                reward += -10  # Time out reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)

        cx, cy = self.cursor_pos

        # Action priority: Sell > Plant/Harvest
        if shift_held:
            if self.harvested_crops > 0:
                # SFX: Cha-ching!
                coins_earned = self.harvested_crops * self.CROP_SELL_PRICE
                self.score += coins_earned
                self._spawn_particles(self.grid_offset_x + self.grid_width, 50, self.COLOR_RIPE, 20, is_coin=True)
                self.harvested_crops = 0
                reward += 1.0
        elif space_held:
            plot_state = self.grid[cy, cx]
            if plot_state == self.STATE_EMPTY:
                if self.score >= self.PLANT_COST:
                    # SFX: Plant seed
                    self.score -= self.PLANT_COST
                    self.grid[cy, cx] = self.STATE_SEED
                    self.growth_timers[cy, cx] = self.GROWTH_STAGE_TIME
                    self._spawn_particles(self.grid_offset_x + cx * 32 + 16, self.grid_offset_y + cy * 32 + 16, self.COLOR_SEED, 10)
                else:
                    reward -= 0.01 # Penalty for trying to plant without money
            elif plot_state == self.STATE_RIPE:
                # SFX: Harvest pop
                self.grid[cy, cx] = self.STATE_EMPTY
                self.growth_timers[cy, cx] = 0
                self.harvested_crops += 1
                reward += 0.1
                self._spawn_particles(self.grid_offset_x + cx * 32 + 16, self.grid_offset_y + cy * 32 + 16, self.COLOR_RIPE, 15)
            else: # Trying to interact with a growing plot
                reward -= 0.01

        return reward

    def _update_crops(self):
        growing_plots = (self.grid > self.STATE_EMPTY) & (self.grid < self.STATE_RIPE)
        self.growth_timers[growing_plots] -= 1
        
        ready_to_advance = (self.growth_timers <= 0) & growing_plots
        self.grid[ready_to_advance] += 1
        self.growth_timers[ready_to_advance] = self.GROWTH_STAGE_TIME

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "harvested_crops": self.harvested_crops,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_game(self):
        cell_size = 32
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(self.grid_offset_x + x * cell_size, self.grid_offset_y + y * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.screen, self.COLOR_SOIL, rect)
                
                state = self.grid[y, x]
                if state == self.STATE_SEED:
                    pygame.draw.circle(self.screen, self.COLOR_SEED, rect.center, 4)
                elif state == self.STATE_GROWING:
                    points = [(rect.centerx, rect.top + 5), (rect.left + 8, rect.bottom - 5), (rect.right - 8, rect.bottom - 5)]
                    pygame.draw.polygon(self.screen, self.COLOR_GROWING, points)
                elif state == self.STATE_RIPE:
                    pygame.draw.circle(self.screen, self.COLOR_RIPE, rect.center, 10)
                    pygame.draw.circle(self.screen, (255,255,255,20), rect.center, 14, 2)

                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Render cursor
        cursor_rect = pygame.Rect(self.grid_offset_x + self.cursor_pos[0] * cell_size,
                                  self.grid_offset_y + self.cursor_pos[1] * cell_size,
                                  cell_size, cell_size)
        
        cursor_surf = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        cursor_surf.fill((*self.COLOR_CURSOR, 80))
        pygame.draw.rect(cursor_surf, self.COLOR_CURSOR, cursor_surf.get_rect(), 2)
        self.screen.blit(cursor_surf, cursor_rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # UI Background
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Score
        score_text = self.font_ui.render(f"COINS: {self.score}/{self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Harvested Crops
        harvest_text = self.font_ui.render(f"HARVEST: {self.harvested_crops}", True, self.COLOR_UI_TEXT)
        self.screen.blit(harvest_text, (self.SCREEN_WIDTH - harvest_text.get_width() - 200, 10))

        # Time Bar
        time_ratio = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
        bar_width = 200
        current_width = int(bar_width * time_ratio)
        time_color = self.COLOR_TIME_BAR if time_ratio > 0.2 else self.COLOR_TIME_BAR_WARN
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, time_color, (10, 10, current_width, 20))
        
        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg_text = "YOU WIN!" if self.win else "TIME'S UP!"
            msg_color = self.COLOR_RIPE if self.win else self.COLOR_TIME_BAR_WARN
            text_surf = self.font_msg.render(msg_text, True, msg_color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _spawn_particles(self, x, y, color, count, is_coin=False):
        for _ in range(count):
            if is_coin:
                dx = random.uniform(-2, 2)
                dy = random.uniform(-5, -2)
            else:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 3)
                dx = math.cos(angle) * speed
                dy = math.sin(angle) * speed
            
            size = random.uniform(3, 7)
            life = random.randint(20, 40)
            self.particles.append(Particle(x, y, color, size, life, dx, dy))

    def validate_implementation(self):
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
    pygame.display.set_caption("Farming Simulator")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    print(GameEnv.user_guide)

    while running:
        # --- Action Mapping for Human Play ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), but our obs is (height, width, 3).
        # We need to transpose it back for display.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(env.FPS)

    pygame.quit()