
# Generated: 2025-08-27T12:37:13.246438
# Source Brief: brief_00100.md
# Brief Index: 100

        
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
        "Controls: Arrow keys to move. Hold Space to collect adjacent crystals. Hold Shift to activate shield."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Explore isometric caverns, collect glowing crystals, and avoid hazards like collapsing floors and falling rocks."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 30, 30
        self.TILE_WIDTH = 32
        self.TILE_HEIGHT = 16

        # Colors
        self.COLOR_BG = (20, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_SHIELD = (100, 180, 255)
        self.COLOR_WALL_TOP = (60, 50, 70)
        self.COLOR_WALL_SIDE = (45, 35, 55)
        self.COLOR_FLOOR = (80, 70, 90)
        self.CRYSTAL_COLORS = [(0, 255, 255), (0, 255, 128), (255, 0, 255), (255, 255, 0)]
        self.HAZARD_COLLAPSE = (100, 80, 110)
        self.HAZARD_WARN = (200, 200, 100)
        self.HAZARD_FALL = (255, 100, 0)
        self.UI_FONT_COLOR = (220, 220, 240)

        # Game parameters
        self.MAX_HEALTH = 100
        self.WIN_CRYSTAL_COUNT = 50
        self.MAX_STEPS = 1000
        self.SHIELD_DURATION = 5
        self.SHIELD_COOLDOWN = 20

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

        # State variables (will be initialized in reset)
        self.player_pos = [0, 0]
        self.health = 0
        self.crystal_count = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = None
        self.crystals = []
        self.hazards = []
        self.shield_active_steps = 0
        self.shield_cooldown_steps = 0
        self.last_player_dist_to_crystal = 0.0
        self.np_random = None

        # Final setup
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.health = self.MAX_HEALTH
        self.crystal_count = 0
        self.game_over = False
        self.shield_active_steps = 0
        self.shield_cooldown_steps = 0

        self._generate_cavern()
        self._place_objects()

        self.last_player_dist_to_crystal = self._find_nearest_crystal_dist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Handle player actions ---
        if shift_pressed and self.shield_cooldown_steps == 0 and self.shield_active_steps == 0:
            self.shield_active_steps = self.SHIELD_DURATION
            self.shield_cooldown_steps = self.SHIELD_COOLDOWN + self.SHIELD_DURATION
            # sfx: shield_activate.wav

        px, py = self.player_pos
        new_px, new_py = px, py
        if movement == 1: new_py -= 1  # Up
        elif movement == 2: new_py += 1  # Down
        elif movement == 3: new_px -= 1  # Left
        elif movement == 4: new_px += 1  # Right

        if 0 <= new_px < self.GRID_WIDTH and 0 <= new_py < self.GRID_HEIGHT and self.grid[new_px, new_py] == 0:
            self.player_pos = [new_px, new_py]

        current_dist = self._find_nearest_crystal_dist()
        if current_dist is not None and self.last_player_dist_to_crystal is not None:
            if current_dist < self.last_player_dist_to_crystal:
                reward += 1.0
            elif current_dist > self.last_player_dist_to_crystal:
                reward -= 0.1
        self.last_player_dist_to_crystal = current_dist

        if space_pressed:
            for crystal in self.crystals[:]:
                cx, cy = crystal['pos']
                if abs(self.player_pos[0] - cx) + abs(self.player_pos[1] - cy) <= 1:
                    self.crystals.remove(crystal)
                    self.crystal_count += 1
                    self.score += 10
                    reward += 10
                    is_near_hazard = any(abs(cx - h['pos'][0]) + abs(cy - h['pos'][1]) <= 1 for h in self.hazards)
                    if is_near_hazard:
                        reward += 2
                        self.score += 2
                    # sfx: crystal_get.wav
                    self.last_player_dist_to_crystal = self._find_nearest_crystal_dist()
                    break

        # --- Update game state ---
        self.steps += 1
        if self.shield_active_steps > 0: self.shield_active_steps -= 1
        if self.shield_cooldown_steps > 0: self.shield_cooldown_steps -= 1

        hazard_update_prob = 0.01 + (self.steps // 200) * 0.01
        for hazard in self.hazards:
            if tuple(hazard['pos']) == tuple(self.player_pos):
                if hazard['type'] == 'collapse' and hazard['state'] == 'armed':
                    hazard['state'] = 'triggered'
                    damage = 20 * (0.5 if self.shield_active_steps > 0 else 1.0)
                    self.health -= damage
                    reward -= 5
                    self.score -= 5
                    self.grid[hazard['pos'][0], hazard['pos'][1]] = 1
                    # sfx: ground_crumble.wav

            if hazard['type'] == 'rock':
                dist_to_player = abs(hazard['pos'][0] - self.player_pos[0]) + abs(hazard['pos'][1] - self.player_pos[1])
                if hazard['state'] == 'idle' and dist_to_player <= 2 and self.np_random.random() < hazard_update_prob:
                    hazard['state'] = 'warn'
                    hazard['timer'] = 3
                    # sfx: rock_warn.wav
                elif hazard['state'] == 'warn':
                    hazard['timer'] -= 1
                    if hazard['timer'] <= 0:
                        hazard['state'] = 'fall'
                        hazard['timer'] = 1
                elif hazard['state'] == 'fall':
                    if tuple(hazard['pos']) == tuple(self.player_pos):
                        damage = 30 * (0.5 if self.shield_active_steps > 0 else 1.0)
                        self.health -= damage
                        reward -= 5
                        self.score -= 5
                        # sfx: rock_impact.wav
                    hazard['state'] = 'cooldown'
                    hazard['timer'] = self.np_random.integers(15, 30)
                elif hazard['state'] == 'cooldown':
                    hazard['timer'] -= 1
                    if hazard['timer'] <= 0:
                        hazard['state'] = 'idle'

        self.health = max(0, self.health)
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.crystal_count >= self.WIN_CRYSTAL_COUNT:
                reward += 100
                self.score += 100
            elif self.health <= 0:
                reward -= 100
                self.score -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "crystals": self.crystal_count,
            "player_pos": self.player_pos,
            "shield_cooldown": self.shield_cooldown_steps,
        }

    def _generate_cavern(self):
        self.grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        x, y = self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2
        self.grid[x, y] = 0
        self.player_pos = [x, y]
        num_floor_tiles = self.np_random.integers(300, 400)
        for _ in range(num_floor_tiles):
            dx, dy = self.np_random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)], p=[0.25, 0.25, 0.25, 0.25])
            x, y = x + dx, y + dy
            if 1 <= x < self.GRID_WIDTH - 1 and 1 <= y < self.GRID_HEIGHT - 1:
                self.grid[x, y] = 0
            else:
                floor_tiles = np.argwhere(self.grid == 0)
                if len(floor_tiles) > 0:
                    x, y = self.np_random.choice(floor_tiles)

    def _place_objects(self):
        self.crystals = []
        self.hazards = []
        floor_tiles = [list(pos) for pos in np.argwhere(self.grid == 0)]
        self.np_random.shuffle(floor_tiles)
        if self.player_pos in floor_tiles:
            floor_tiles.remove(self.player_pos)

        num_crystals = self.WIN_CRYSTAL_COUNT + 10
        for _ in range(min(num_crystals, len(floor_tiles))):
            pos = floor_tiles.pop(0)
            color_tuple = self.np_random.choice(self.CRYSTAL_COLORS)
            self.crystals.append({'pos': pos, 'color': tuple(color_tuple)})

        hazard_freq = 0.1
        num_hazards = int(len(floor_tiles) * hazard_freq)
        for _ in range(min(num_hazards, len(floor_tiles))):
            pos = floor_tiles.pop(0)
            hazard_type = self.np_random.choice(['collapse', 'rock'])
            if hazard_type == 'collapse':
                self.hazards.append({'type': 'collapse', 'pos': pos, 'state': 'armed'})
            else:
                self.hazards.append({'type': 'rock', 'pos': pos, 'state': 'idle', 'timer': 0})

    def _check_termination(self):
        return self.health <= 0 or self.crystal_count >= self.WIN_CRYSTAL_COUNT or self.steps >= self.MAX_STEPS

    def _find_nearest_crystal_dist(self):
        if not self.crystals:
            return None
        player_pos_np = np.array(self.player_pos)
        crystal_positions = np.array([c['pos'] for c in self.crystals])
        distances = np.sum(np.abs(crystal_positions - player_pos_np), axis=1)
        return np.min(distances) if distances.size > 0 else None

    def _world_to_screen(self, x, y):
        screen_x = (x - y) * self.TILE_WIDTH / 2 + self.SCREEN_WIDTH / 2
        screen_y = (x + y) * self.TILE_HEIGHT / 2 + self.SCREEN_HEIGHT / 4 - self.GRID_HEIGHT / 2 * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)

    def _render_game(self):
        render_queue = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0:
                    render_queue.append(('floor', (x, y), x + y))
                else:
                    is_edge = any(0 <= x+dx < self.GRID_WIDTH and 0 <= y+dy < self.GRID_HEIGHT and self.grid[x+dx, y+dy] == 0 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)])
                    if is_edge:
                        render_queue.append(('wall', (x, y), x + y))

        for hazard in self.hazards:
            pos = hazard['pos']
            depth = pos[0] + pos[1]
            if hazard['type'] == 'collapse' and hazard['state'] == 'armed':
                render_queue.append(('hazard_collapse', pos, depth + 0.1))
            elif hazard['type'] == 'rock':
                if hazard['state'] == 'warn':
                    render_queue.append(('hazard_warn', pos, depth + 0.7))
                elif hazard['state'] == 'fall':
                    render_queue.append(('hazard_fall', pos, depth + 0.8))

        for crystal in self.crystals:
            pos = crystal['pos']
            render_queue.append(('crystal', crystal, pos[0] + pos[1] + 0.5))

        render_queue.append(('player', self.player_pos, self.player_pos[0] + self.player_pos[1] + 0.6))
        render_queue.sort(key=lambda item: item[2])

        for item_type, data, _ in render_queue:
            pos = data if item_type in ['floor', 'wall', 'player', 'hazard_collapse', 'hazard_warn', 'hazard_fall'] else data['pos']
            if item_type == 'floor': self._draw_iso_cube(pos[0], pos[1], self.COLOR_FLOOR, False)
            elif item_type == 'wall': self._draw_iso_cube(pos[0], pos[1], self.COLOR_WALL_TOP, True)
            elif item_type == 'hazard_collapse': self._draw_iso_cube(pos[0], pos[1], self.HAZARD_COLLAPSE, False)
            elif item_type == 'crystal': self._draw_crystal(pos[0], pos[1], data['color'])
            elif item_type == 'player': self._draw_player(pos[0], pos[1])
            elif item_type == 'hazard_warn': self._draw_hazard_warn(pos[0], pos[1])
            elif item_type == 'hazard_fall': self._draw_hazard_fall(pos[0], pos[1])

    def _draw_iso_cube(self, x, y, color, is_wall=False):
        sx, sy = self._world_to_screen(x, y)
        h, w = self.TILE_HEIGHT, self.TILE_WIDTH
        top_points = [(sx, sy), (sx + w / 2, sy + h / 2), (sx, sy + h), (sx - w / 2, sy + h / 2)]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, color)
        pygame.gfxdraw.aapolygon(self.screen, top_points, color)
        if is_wall:
            side_color = self.COLOR_WALL_SIDE
            left_points = [(sx - w / 2, sy + h / 2), (sx, sy + h), (sx, sy + h + h), (sx - w / 2, sy + h / 2 + h)]
            pygame.gfxdraw.filled_polygon(self.screen, left_points, side_color)
            pygame.gfxdraw.aapolygon(self.screen, left_points, side_color)
            right_points = [(sx + w / 2, sy + h / 2), (sx, sy + h), (sx, sy + h + h), (sx + w / 2, sy + h / 2 + h)]
            pygame.gfxdraw.filled_polygon(self.screen, right_points, side_color)
            pygame.gfxdraw.aapolygon(self.screen, right_points, side_color)

    def _draw_player(self, x, y):
        sx, sy = self._world_to_screen(x, y)
        rect = pygame.Rect(sx - self.TILE_WIDTH * 0.2, sy - self.TILE_HEIGHT, self.TILE_WIDTH * 0.4, self.TILE_HEIGHT * 1.5)
        pygame.draw.ellipse(self.screen, self.COLOR_PLAYER, rect)
        if self.shield_active_steps > 0:
            shield_surface = pygame.Surface((self.TILE_WIDTH * 2, self.TILE_WIDTH * 2), pygame.SRCALPHA)
            alpha = 90 + math.sin(self.steps * 0.5) * 30
            radius = int(self.TILE_WIDTH * 0.6 + math.sin(self.steps * 0.5) * 2)
            pygame.gfxdraw.filled_circle(shield_surface, self.TILE_WIDTH, self.TILE_WIDTH, radius, (*self.COLOR_SHIELD, alpha))
            pygame.gfxdraw.aacircle(shield_surface, self.TILE_WIDTH, self.TILE_WIDTH, radius, (*self.COLOR_SHIELD, alpha * 1.5))
            self.screen.blit(shield_surface, (sx - self.TILE_WIDTH, sy - self.TILE_WIDTH + self.TILE_HEIGHT / 2))

    def _draw_crystal(self, x, y, color):
        sx, sy = self._world_to_screen(x, y)
        y_offset = self.TILE_HEIGHT / 2
        glow_radius = int(self.TILE_WIDTH * 0.5 + math.sin(self.steps * 0.2 + x) * 3)
        glow_alpha = 50 + math.sin(self.steps * 0.2 + y) * 20
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surface, glow_radius, glow_radius, glow_radius, (*color, glow_alpha))
        self.screen.blit(glow_surface, (sx - glow_radius, sy - glow_radius + y_offset))
        base_size = self.TILE_WIDTH / 4
        points = [(sx, sy - base_size * 0.5 + y_offset), (sx + base_size * 0.5, sy + y_offset), (sx, sy + base_size * 0.5 + y_offset), (sx - base_size * 0.5, sy + y_offset)]
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        pygame.gfxdraw.aapolygon(self.screen, points, (255, 255, 255))

    def _draw_hazard_warn(self, x, y):
        sx, sy = self._world_to_screen(x, y)
        if self.steps % 4 < 2:
            radius = int(self.TILE_WIDTH / 3)
            pygame.gfxdraw.filled_circle(self.screen, sx, sy + self.TILE_HEIGHT // 2, radius, self.HAZARD_WARN)
            pygame.gfxdraw.aacircle(self.screen, sx, sy + self.TILE_HEIGHT // 2, radius, self.HAZARD_WARN)

    def _draw_hazard_fall(self, x, y):
        sx, sy = self._world_to_screen(x, y)
        rect = pygame.Rect(sx - self.TILE_WIDTH * 0.3, sy - self.TILE_HEIGHT, self.TILE_WIDTH * 0.6, self.TILE_HEIGHT * 1.5)
        pygame.draw.ellipse(self.screen, self.HAZARD_FALL, rect)

    def _render_ui(self):
        health_text = self.font_large.render(f"Health: {int(self.health)}", True, self.UI_FONT_COLOR)
        self.screen.blit(health_text, (10, 10))
        crystal_text = self.font_large.render(f"Crystals: {self.crystal_count}/{self.WIN_CRYSTAL_COUNT}", True, self.UI_FONT_COLOR)
        self.screen.blit(crystal_text, (self.SCREEN_WIDTH - crystal_text.get_width() - 10, 10))
        if self.shield_active_steps > 0:
            shield_text, color = f"Shield: {self.shield_active_steps}", (150, 220, 255)
        elif self.shield_cooldown_steps > 0:
            shield_text, color = f"Cooldown: {self.shield_cooldown_steps}", (150, 150, 150)
        else:
            shield_text, color = "Shield Ready", self.UI_FONT_COLOR
        self.screen.blit(self.font_small.render(shield_text, True, color), (10, 45))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")