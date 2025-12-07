
# Generated: 2025-08-27T19:02:58.388649
# Source Brief: brief_02034.md
# Brief Index: 2034

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to plant a seed on empty soil. "
        "Press Shift over a ripe crop to harvest, or over the barn to sell all harvested crops."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced farming simulator. Plant, grow, and harvest crops, then sell them at the barn. "
        "Reach the profit goal of $1000 before time runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (20, 30, 25)
    COLOR_FIELD = (101, 67, 33)
    COLOR_GRID = (81, 47, 13)
    COLOR_BARN = (180, 40, 40)
    COLOR_BARN_ROOF = (140, 30, 30)
    COLOR_CURSOR = (220, 255, 255, 150) # RGBA for transparency
    COLOR_CROP_GROWING = (60, 180, 75)
    COLOR_CROP_RIPE = (255, 225, 25)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_UI_BG = (10, 15, 12, 180) # RGBA

    # Grid
    GRID_WIDTH = 15
    GRID_HEIGHT = 8
    TILE_SIZE = 40
    FIELD_X_OFFSET = 40
    FIELD_Y_OFFSET = 60

    # Game Rules
    TIME_LIMIT_SECONDS = 180
    MAX_STEPS = TIME_LIMIT_SECONDS * FPS
    WIN_PROFIT = 1000
    CROP_GROWTH_SECONDS = 5
    CROP_GROWTH_FRAMES = CROP_GROWTH_SECONDS * FPS
    CROP_SELL_PRICE = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_pop = pygame.font.SysFont("Arial", 18, bold=True)

        # Game state variables are initialized in reset()
        self.cursor_pos = None
        self.field = None
        self.harvested_crops = None
        self.profit = None
        self.time_remaining = None
        self.steps = None
        self.particles = None
        self.game_over = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.profit = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.harvested_crops = 0

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.field = [
            [{'state': 'empty', 'growth': 0} for _ in range(self.GRID_WIDTH)]
            for _ in range(self.GRID_HEIGHT)
        ]
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Update state from previous frame ---
        self._update_crops()
        self._update_particles()
        self.time_remaining = max(0, self.time_remaining - 1.0 / self.FPS)
        self.steps += 1
        reward = -0.01 # Small penalty for time passing

        # --- Process Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Get tile under cursor
        tile_x, tile_y = self.cursor_pos
        tile = self.field[tile_y][tile_x]

        # Action: Plant (Space)
        if space_held and tile['state'] == 'empty':
            tile['state'] = 'growing'
            tile['growth'] = 0
            # Sound: plant_seed.wav
            self._add_particles(self.cursor_pos, 5, self.COLOR_FIELD, 0.5)

        # Action: Harvest/Sell (Shift)
        if shift_held:
            # Check if cursor is over the barn area
            if tile_x == 0 and tile_y < 3: # Barn is at column 0, top 3 rows
                if self.harvested_crops > 0:
                    money_earned = self.harvested_crops * self.CROP_SELL_PRICE
                    self.profit += money_earned
                    reward += 1.0 # Reward for selling
                    # Sound: cha_ching.wav
                    self._add_particles(self.cursor_pos, self.harvested_crops, (50, 200, 50), 1.5, "$")
                    self.harvested_crops = 0
            # Check if cursor is over a ripe crop
            elif tile['state'] == 'ripe':
                tile['state'] = 'empty'
                tile['growth'] = 0
                self.harvested_crops += 1
                reward += 0.1 # Reward for harvesting
                # Sound: harvest_pop.wav
                self._add_particles(self.cursor_pos, 10, self.COLOR_CROP_RIPE, 0.7)

        # --- Check Termination ---
        terminated = False
        if self.profit >= self.WIN_PROFIT:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_crops(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile = self.field[y][x]
                if tile['state'] == 'growing':
                    tile['growth'] += 1
                    if tile['growth'] >= self.CROP_GROWTH_FRAMES:
                        tile['state'] = 'ripe'

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.profit,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "harvested_crops": self.harvested_crops
        }

    def _render_game(self):
        # Draw field background
        field_rect = pygame.Rect(self.FIELD_X_OFFSET, self.FIELD_Y_OFFSET,
                                 self.GRID_WIDTH * self.TILE_SIZE,
                                 self.GRID_HEIGHT * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_FIELD, field_rect)

        # Draw grid and crops
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile = self.field[y][x]
                rect = pygame.Rect(self.FIELD_X_OFFSET + x * self.TILE_SIZE,
                                   self.FIELD_Y_OFFSET + y * self.TILE_SIZE,
                                   self.TILE_SIZE, self.TILE_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                center_x, center_y = rect.center
                if tile['state'] == 'growing':
                    radius = int((self.TILE_SIZE / 2 - 4) * (tile['growth'] / self.CROP_GROWTH_FRAMES))
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, max(0, radius), self.COLOR_CROP_GROWING)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, max(0, radius), self.COLOR_CROP_GROWING)
                elif tile['state'] == 'ripe':
                    radius = int(self.TILE_SIZE / 2 - 4)
                    pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_CROP_RIPE)
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_CROP_RIPE)

        # Draw Barn
        barn_x = self.FIELD_X_OFFSET
        barn_y = self.FIELD_Y_OFFSET
        barn_w = self.TILE_SIZE
        barn_h = self.TILE_SIZE * 3
        pygame.draw.rect(self.screen, self.COLOR_BARN, (barn_x, barn_y, barn_w, barn_h))
        pygame.draw.polygon(self.screen, self.COLOR_BARN_ROOF, [(barn_x, barn_y), (barn_x + barn_w, barn_y), (barn_x + barn_w/2, barn_y - 20)])
        # Barn door
        pygame.draw.rect(self.screen, self.COLOR_BARN_ROOF, (barn_x + 10, barn_y + barn_h - 25, barn_w - 20, 20))


        # Draw Cursor
        cursor_x = self.FIELD_X_OFFSET + self.cursor_pos[0] * self.TILE_SIZE
        cursor_y = self.FIELD_Y_OFFSET + self.cursor_pos[1] * self.TILE_SIZE
        cursor_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surface, self.COLOR_CURSOR, cursor_surface.get_rect(), 4, border_radius=4)
        self.screen.blit(cursor_surface, (cursor_x, cursor_y))

    def _render_ui(self):
        # UI Background Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 50), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Profit
        profit_text = self.font_ui.render(f"Profit: ${self.profit} / ${self.WIN_PROFIT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(profit_text, (15, 12))

        # Time
        time_text = self.font_ui.render(f"Time: {int(self.time_remaining)}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 15, 12))

        # Harvested Crops
        crop_text = self.font_ui.render(f"Harvested: {self.harvested_crops}", True, self.COLOR_UI_TEXT)
        self.screen.blit(crop_text, (self.SCREEN_WIDTH // 2 - crop_text.get_width() // 2, 12))

    def _add_particles(self, tile_pos, count, color, life_sec, text=None):
        center_x = self.FIELD_X_OFFSET + tile_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        center_y = self.FIELD_Y_OFFSET + tile_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = int(life_sec * self.FPS)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': life, 'color': color, 'text': text})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / (self.FPS * 2)))
            if p['text']:
                pop_text = self.font_pop.render(p['text'], True, p['color'])
                pop_text.set_alpha(max(0, alpha))
                self.screen.blit(pop_text, (int(p['pos'][0]), int(p['pos'][1])))
            else:
                size = int(5 * (p['life'] / (self.FPS * 1)))
                if size > 0:
                    pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Setup for human play
    pygame.display.set_caption("Farming Simulator")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("      MANUAL PLAY MODE")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate Control ---
        clock.tick(GameEnv.FPS)

    pygame.quit()