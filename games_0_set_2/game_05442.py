import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to plant seeds. Shift to water, harvest, or sell."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced farming sim. Plant, water, and harvest crops to reach $1000 before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    TIME_LIMIT = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (40, 50, 40)
    COLOR_GRID = (60, 70, 60)
    COLOR_SOIL = (100, 80, 60)
    COLOR_SOIL_WATERED = (60, 45, 35)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_CURSOR_INVALID = (255, 50, 50)
    COLOR_TEXT = (250, 250, 220)
    COLOR_WELL_OUTLINE = (100, 100, 100)
    COLOR_WELL_WATER = (50, 100, 200)
    COLOR_MARKET_RED = (220, 50, 50)
    COLOR_MARKET_WHITE = (255, 255, 255)

    # Crop Colors
    CROP_COLORS = {
        1: (140, 100, 80),  # Seed
        2: (50, 200, 50),  # Seedling
        3: (100, 255, 100),  # Mature
    }

    # Game Grid
    GRID_SIZE = 10
    GRID_ORIGIN_X, GRID_ORIGIN_Y = 120, 50
    TILE_SIZE = 30

    # Special Locations (in grid coordinates)
    WELL_POS = (1, -1)
    MARKET_POS = (GRID_SIZE - 2, -1)

    # Game Parameters
    WIN_SCORE = 1000
    MAX_WATER = 50
    INITIAL_WATER = 30
    WATER_REFILL_RATE = 15  # steps per unit
    CROP_SELL_PRICE = 20
    GROWTH_THRESHOLD = 90  # steps of being watered to grow

    # Action durations (in steps)
    ACTION_DURATIONS = {
        'plant': 5,
        'water': 2,
        'harvest': 3,
        'sell': 10,
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)

        self.render_mode = render_mode
        self.steps_since_action = 0

        # self.reset() is called here, but we need to initialize attributes first
        self.particles = []
        self.floating_texts = []

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_remaining = self.TIME_LIMIT
        self.game_over = False

        self.grid = np.array([
            [{'stage': 0, 'growth': 0, 'watered': False} for _ in range(self.GRID_SIZE)]
            for _ in range(self.GRID_SIZE)
        ])

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.water = self.INITIAL_WATER
        self.harvested_produce = 0

        self.player_action = None
        self.player_action_timer = 0

        self.particles = []
        self.floating_texts = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0

        self._update_world_state()
        action_reward = self._update_player_action()
        reward += action_reward
        self._handle_input(action)
        self._update_effects()

        self.steps += 1
        self.time_remaining -= 1

        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.player_action_timer > 0:
            return

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)

        # --- Actions (prioritize Shift over Space) ---
        cursor_x, cursor_y = self.cursor_pos
        tile = self.grid[cursor_y, cursor_x]

        if shift_pressed:
            # Sell
            if self.cursor_pos == [self.MARKET_POS[0], self.MARKET_POS[1] + 1] and self.harvested_produce > 0:
                self._start_action('sell')
            # Harvest
            elif tile['stage'] == 3:
                self._start_action('harvest')
            # Water
            elif 0 < tile['stage'] < 3 and not tile['watered'] and self.water > 0:
                self._start_action('water')
        elif space_pressed:
            # Plant
            if tile['stage'] == 0: # No water cost to plant seeds
                self._start_action('plant')


    def _start_action(self, action_name):
        self.player_action = action_name
        self.player_action_timer = self.ACTION_DURATIONS[action_name]

        # Deduct costs immediately
        if action_name == 'water':
            self.water -= 1
            # sfx: water_splash
            self._add_particles(self.cursor_pos, 20, (50, 100, 200), 0.5)
        elif action_name == 'plant':
            # sfx: plant_seed
            self._add_particles(self.cursor_pos, 10, self.COLOR_SOIL, 0.3)


    def _update_player_action(self):
        if self.player_action_timer > 0:
            self.player_action_timer -= 1
            if self.player_action_timer == 0:
                return self._complete_action()
        return 0

    def _complete_action(self):
        reward = 0
        action = self.player_action
        self.player_action = None

        cursor_x, cursor_y = self.cursor_pos
        tile = self.grid[cursor_y, cursor_x]

        if action == 'plant':
            tile['stage'] = 1
        elif action == 'water':
            tile['watered'] = True
            reward += 0.1
        elif action == 'harvest':
            tile['stage'] = 0
            tile['growth'] = 0
            tile['watered'] = False
            self.harvested_produce += 1
            reward += 0.2
            # sfx: harvest_pop
            self._add_particles(self.cursor_pos, 30, self.CROP_COLORS[3], 0.8)
        elif action == 'sell':
            amount_sold = self.harvested_produce
            profit = amount_sold * self.CROP_SELL_PRICE
            self.score += profit
            self.harvested_produce = 0
            reward += 1.0 * amount_sold
            # sfx: cash_register
            self._add_floating_text(f"+${profit}", (
            self.GRID_ORIGIN_X + self.MARKET_POS[0] * self.TILE_SIZE, self.GRID_ORIGIN_Y - 10), (255, 223, 0))
            self._add_coin_particles((self.GRID_ORIGIN_X + self.MARKET_POS[0] * self.TILE_SIZE, self.GRID_ORIGIN_Y - 10),
                                     30)

        return reward

    def _update_world_state(self):
        # Water refill
        if self.steps > 0 and self.steps % self.WATER_REFILL_RATE == 0:
            self.water = min(self.MAX_WATER, self.water + 1)

        # Crop growth
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                tile = self.grid[y, x]
                if tile['stage'] > 0 and tile['stage'] < 3 and tile['watered']:
                    tile['growth'] += 1
                    if tile['growth'] >= self.GROWTH_THRESHOLD:
                        tile['stage'] += 1
                        tile['growth'] = 0
                        tile['watered'] = False
                        # sfx: grow_ding
                        self._add_particles([x, y], 15, self.CROP_COLORS[tile['stage']], 0.6)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            if not self.game_over:
                self._add_floating_text("YOU WIN!", (self.WIDTH // 2, self.HEIGHT // 2), (100, 255, 100), size=2)
            return True, 100.0
        if self.time_remaining <= 0:
            if not self.game_over:
                self._add_floating_text("TIME UP!", (self.WIDTH // 2, self.HEIGHT // 2), (255, 100, 100), size=2)
            return True, -100.0
        return False, 0.0

    def _update_effects(self):
        # Update and filter particles
        next_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] - 0.1)
            if p['life'] > 0 and p['radius'] > 0:
                next_particles.append(p)
        self.particles = next_particles

        # Update and filter floating texts
        next_floating_texts = []
        for ft in self.floating_texts:
            ft['pos'][1] -= ft['vel']
            ft['life'] -= 1
            if ft['life'] > 0:
                next_floating_texts.append(ft)
        self.floating_texts = next_floating_texts

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_world_objects()
        self._render_grid_and_crops()
        self._render_cursor()
        self._render_effects()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_world_objects(self):
        # Well
        well_cx = self.GRID_ORIGIN_X + self.WELL_POS[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        well_cy = self.GRID_ORIGIN_Y + self.WELL_POS[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        pygame.draw.circle(self.screen, self.COLOR_WELL_OUTLINE, (well_cx, well_cy), 20, 5)
        pygame.draw.circle(self.screen, self.COLOR_WELL_WATER, (well_cx, well_cy), 15)

        # Market
        market_x = self.GRID_ORIGIN_X + self.MARKET_POS[0] * self.TILE_SIZE
        market_y = self.GRID_ORIGIN_Y + self.MARKET_POS[1] * self.TILE_SIZE
        pygame.draw.rect(self.screen, (139, 69, 19), (market_x - 5, market_y, self.TILE_SIZE * 2 + 10, self.TILE_SIZE))
        # Awning
        for i in range(7): # Extend awning to cover full width
            color = self.COLOR_MARKET_RED if i % 2 == 0 else self.COLOR_MARKET_WHITE
            pygame.draw.rect(self.screen, color, (market_x - 5 + i * 10, market_y - 10, 10, 10))

    def _render_grid_and_crops(self):
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                tile = self.grid[y, x]
                rect = pygame.Rect(
                    self.GRID_ORIGIN_X + x * self.TILE_SIZE,
                    self.GRID_ORIGIN_Y + y * self.TILE_SIZE,
                    self.TILE_SIZE, self.TILE_SIZE
                )

                # Draw soil
                soil_color = self.COLOR_SOIL_WATERED if tile['watered'] else self.COLOR_SOIL
                pygame.draw.rect(self.screen, soil_color, rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                # Draw crop
                if tile['stage'] > 0:
                    cx, cy = rect.center
                    color = self.CROP_COLORS[tile['stage']]
                    if tile['stage'] == 1:  # Seed
                        pygame.gfxdraw.filled_circle(self.screen, cx, cy, 3, color)
                    elif tile['stage'] == 2:  # Seedling
                        pygame.gfxdraw.filled_circle(self.screen, cx, cy - 2, 4, color)
                        pygame.draw.line(self.screen, color, (cx, cy), (cx, cy + 4), 2)
                    elif tile['stage'] == 3:  # Mature
                        pygame.gfxdraw.filled_circle(self.screen, cx, cy - 5, 8, color)
                        pygame.gfxdraw.filled_circle(self.screen, cx - 6, cy, 6, color)
                        pygame.gfxdraw.filled_circle(self.screen, cx + 6, cy, 6, color)

    def _render_cursor(self):
        cursor_x, cursor_y = self.cursor_pos
        is_special_loc = (cursor_y == self.MARKET_POS[1] + 1 and self.MARKET_POS[0] <= cursor_x <= self.MARKET_POS[0] + 1)

        if is_special_loc:
            rect = pygame.Rect(
                self.GRID_ORIGIN_X + self.MARKET_POS[0] * self.TILE_SIZE - 5,
                self.GRID_ORIGIN_Y + (self.MARKET_POS[1] + 1) * self.TILE_SIZE,
                self.TILE_SIZE * 2 + 10, self.TILE_SIZE
            )
        else:
            rect = pygame.Rect(
                self.GRID_ORIGIN_X + cursor_x * self.TILE_SIZE,
                self.GRID_ORIGIN_Y + cursor_y * self.TILE_SIZE,
                self.TILE_SIZE, self.TILE_SIZE
            )

        # Pulsating effect
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        line_width = 2 + int(pulse * 2)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, line_width)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"${self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Time
        time_sec = self.time_remaining // self.FPS
        time_text = self.font_medium.render(f"Time: {time_sec}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 20))

        # Water Bar
        water_bar_bg = pygame.Rect(self.GRID_ORIGIN_X - 40, self.GRID_ORIGIN_Y, 20, self.GRID_SIZE * self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, water_bar_bg, 2)
        water_ratio = self.water / self.MAX_WATER
        water_fill_height = (water_bar_bg.height - 4) * water_ratio
        water_bar_fill = pygame.Rect(
            water_bar_bg.x + 2,
            water_bar_bg.y + water_bar_bg.height - water_fill_height - 2,
            water_bar_bg.width - 4,
            water_fill_height
        )
        pygame.draw.rect(self.screen, self.COLOR_WELL_WATER, water_bar_fill)

        # Harvested Produce
        if self.harvested_produce > 0:
            harvest_icon_rect = pygame.Rect(self.WIDTH - 150, self.HEIGHT - 40, 30, 30)
            cx, cy = harvest_icon_rect.center
            pygame.gfxdraw.filled_circle(self.screen, cx, cy - 2, 10, self.CROP_COLORS[3])
            harvest_text = self.font_medium.render(f"x {self.harvested_produce}", True, self.COLOR_TEXT)
            self.screen.blit(harvest_text,
                             (harvest_icon_rect.right + 5, harvest_icon_rect.centery - harvest_text.get_height() // 2))

        # Action bar
        if self.player_action_timer > 0:
            ratio = self.player_action_timer / self.ACTION_DURATIONS[self.player_action]
            bar_width = 100
            bar_y = self.HEIGHT - 25
            pygame.draw.rect(self.screen, self.COLOR_GRID, ((self.WIDTH - bar_width) // 2, bar_y, bar_width, 15), 2)
            fill_width = (bar_width - 4) * (1 - ratio)
            pygame.draw.rect(self.screen, self.COLOR_CURSOR,
                             ((self.WIDTH - bar_width) // 2 + 2, bar_y + 2, fill_width, 11))

    def _render_effects(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

        for ft in self.floating_texts:
            alpha = min(255, ft['life'] * 20)
            font = self.font_large if ft['size'] == 2 else self.font_medium
            text_surf = font.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf,
                             (ft['pos'][0] - text_surf.get_width() // 2, ft['pos'][1] - text_surf.get_height() // 2))

    def _add_particles(self, grid_pos, count, color, speed_mult):
        px = self.GRID_ORIGIN_X + grid_pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        py = self.GRID_ORIGIN_Y + grid_pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(15, 30),
                'radius': random.uniform(2, 5),
                'color': color
            })

    def _add_coin_particles(self, pos, count):
        for _ in range(count):
            angle = random.uniform(-math.pi * 0.75, -math.pi * 0.25)
            speed = random.uniform(2, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'radius': random.uniform(3, 6),
                'color': (255, 223, 0)
            })

    def _add_floating_text(self, text, pos, color, life=45, vel=0.5, size=1):
        self.floating_texts.append({
            'text': text,
            'pos': list(pos),
            'color': color,
            'life': life,
            'vel': vel,
            'size': size
        })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "water": self.water,
            "harvested_produce": self.harvested_produce,
            "cursor_pos": self.cursor_pos,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)

        # Test observation space
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # We need a display for manual play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Farming Simulator")

    terminated = False
    total_reward = 0

    # Game loop for manual play
    while not terminated:
        # --- Action mapping for human players ---
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Print info ---
        if env.steps % 30 == 0:
            print(
                f"Step: {info['steps']}, Score: {info['score']}, Reward: {total_reward:.2f}, Time: {info['time_remaining'] // env.FPS}")

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()