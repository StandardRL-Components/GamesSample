
# Generated: 2025-08-28T06:34:00.174387
# Source Brief: brief_02955.md
# Brief Index: 2955

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place a defensive block."
    )

    game_description = (
        "Defend your base by building a maze of blocks. Survive 20 waves of enemies that get progressively stronger."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 16
        self.MAX_STEPS = 1000
        self.WIN_WAVE = 20

        # Isometric projection constants
        self.TILE_WIDTH_HALF = 20
        self.TILE_HEIGHT_HALF = 10
        self.ISO_OFFSET_X = self.WIDTH // 2
        self.ISO_OFFSET_Y = 80

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_BLOCK = (100, 180, 255)
        self.COLOR_BLOCK_TOP = (150, 220, 255)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_ENEMY_SHADOW = (20, 20, 20, 150)
        self.COLOR_BASE = (70, 200, 120)
        self.COLOR_BASE_TOP = (120, 255, 170)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_CURSOR = (255, 255, 255, 100)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_GREEN = (0, 200, 0)
        self.COLOR_HEALTH_RED = (200, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # --- Game State ---
        self.np_random = None
        self.grid = None
        self.base_pos = None
        self.base_health = None
        self.max_base_health = None
        self.cursor_pos = None
        self.enemies = None
        self.projectiles = None
        self.particles = None
        self.wave = None
        self.wave_cooldown = None
        self.enemies_to_spawn_this_wave = None
        self.spawn_timer = None
        self.turret_cooldown = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.base_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT - 3)
        self.grid[self.base_pos] = 2 # Mark base location
        self.max_base_health = 50
        self.base_health = self.max_base_health
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave = 0
        self.wave_cooldown = 0
        self.enemies_to_spawn_this_wave = []
        self.spawn_timer = 0
        self.turret_cooldown = 0
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self._start_new_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # 1. Player Action
        movement, space_press, _ = action
        self._handle_player_action(movement, space_press == 1)

        # 2. Game Logic Update
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                reward += self._start_new_wave()
        else:
            spawn_reward = self._update_spawner()
            reward += spawn_reward

        self._update_turret()
        hit_reward = self._update_projectiles()
        reward += hit_reward
        damage_penalty, enemies_cleared = self._update_enemies()
        reward += damage_penalty
        
        self._update_particles()

        if enemies_cleared and not self.enemies_to_spawn_this_wave:
            self.wave_cooldown = 90 # 3 seconds at 30fps

        # 3. Check Termination
        terminated = False
        if self.base_health <= 0:
            reward = -100
            self.game_over = True
            terminated = True
        elif self.wave > self.WIN_WAVE:
            reward += 200 # Final win bonus
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_action(self, movement, place_block):
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # Place block
        cursor_tuple = tuple(self.cursor_pos)
        if place_block and self.grid[cursor_tuple] == 0:
            self.grid[cursor_tuple] = 1 # 1 for player block
            # Sound: block_place.wav

    def _start_new_wave(self):
        self.wave += 1
        if self.wave > self.WIN_WAVE:
            return 0
            
        # Difficulty scaling
        num_enemies = self.wave
        enemy_health = 1 + math.floor(self.wave * 0.1)
        spawn_interval = max(30, 60 - self.wave)

        self.enemies_to_spawn_this_wave = [(enemy_health, spawn_interval) for _ in range(num_enemies)]
        self.spawn_timer = 0
        
        # Sound: wave_start.wav
        return 100 if self.wave > 1 else 0 # Reward for surviving a wave

    def _update_spawner(self):
        if not self.enemies_to_spawn_this_wave:
            return 0
            
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            health, interval = self.enemies_to_spawn_this_wave.pop(0)
            self.spawn_timer = interval
            
            spawn_points = [
                (0, self.np_random.integers(0, self.GRID_HEIGHT)),
                (self.GRID_WIDTH - 1, self.np_random.integers(0, self.GRID_HEIGHT)),
                (self.np_random.integers(0, self.GRID_WIDTH), 0)
            ]
            spawn_pos = list(random.choice(spawn_points))
            
            if self.grid[tuple(spawn_pos)] == 0:
                self.enemies.append({
                    "pos": spawn_pos,
                    "health": health,
                    "max_health": health,
                    "path": deque(),
                    "bob": self.np_random.uniform(0, 2 * math.pi)
                })
                # Sound: enemy_spawn.wav
        return 0

    def _update_turret(self):
        self.turret_cooldown -= 1
        if self.turret_cooldown <= 0 and self.enemies:
            # Find closest enemy
            closest_enemy = min(self.enemies, key=lambda e: self._dist_sq(e["pos"], self.base_pos))
            
            self.turret_cooldown = 15 # Fire rate
            
            start_pos = self._cart_to_iso(*self.base_pos)
            end_pos = self._cart_to_iso(*closest_enemy["pos"])
            
            angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            speed = 8
            
            self.projectiles.append({
                "pos": list(start_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 100
            })
            # Sound: turret_fire.wav

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]
            proj["life"] -= 1

            if proj["life"] <= 0:
                self.projectiles.remove(proj)
                continue

            for enemy in self.enemies[:]:
                enemy_screen_pos = self._cart_to_iso(*enemy["pos"])
                if self._dist_sq(proj["pos"], enemy_screen_pos) < 10**2:
                    # Hit!
                    enemy["health"] -= 1
                    self._create_particles(proj["pos"], self.COLOR_ENEMY, 5)
                    # Sound: enemy_hit.wav
                    if enemy["health"] <= 0:
                        self.enemies.remove(enemy)
                        self._create_particles(enemy_screen_pos, self.COLOR_ENEMY, 20)
                        reward += 10 # Destroyed enemy reward
                        # Sound: enemy_explode.wav
                    
                    if proj in self.projectiles:
                        self.projectiles.remove(proj)
                    break
        return reward

    def _update_enemies(self):
        damage_penalty = 0
        for enemy in self.enemies:
            enemy["bob"] = (enemy["bob"] + 0.1) % (2 * math.pi)
            
            # Manhattan distance to base
            dist = abs(enemy["pos"][0] - self.base_pos[0]) + abs(enemy["pos"][1] - self.base_pos[1])
            
            if dist <= 1:
                # Attack base
                self.base_health -= 1
                damage_penalty -= 1
                self._create_particles(self._cart_to_iso(*self.base_pos), self.COLOR_BASE, 10)
                # Sound: base_damage.wav
            else:
                # Move
                if not enemy["path"]:
                    path = self._pathfind(tuple(enemy["pos"]), self.base_pos)
                    if path:
                        enemy["path"] = deque(path[1:]) # remove start pos

                if enemy["path"]:
                    next_pos = enemy["path"].popleft()
                    enemy["pos"] = list(next_pos)
        
        enemies_cleared = not self.enemies
        return damage_penalty, enemies_cleared

    def _pathfind(self, start, end):
        q = deque([(start, [start])])
        visited = {start}
        
        while q:
            (vx, vy), path = q.popleft()
            if (vx, vy) == end:
                return path

            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(neighbors) # Add randomness to pathing
            for dx, dy in neighbors:
                nx, ny = vx + dx, vy + dy
                if (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and
                        (nx, ny) not in visited and self.grid[nx, ny] != 1):
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    q.append(((nx, ny), new_path))
        return None # No path found

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.95
            p["vel"][1] *= 0.95
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Draw grid floor
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._cart_to_iso(x, y)
                p2 = self._cart_to_iso(x + 1, y)
                p3 = self._cart_to_iso(x + 1, y + 1)
                p4 = self._cart_to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # 2. Collect and sort all drawable grid objects
        renderables = []
        # Add blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 1:
                    renderables.append({"type": "block", "pos": (x, y), "sort_key": x + y})
        # Add base
        renderables.append({"type": "base", "pos": self.base_pos, "sort_key": self.base_pos[0] + self.base_pos[1]})
        # Add enemies
        for enemy in self.enemies:
            renderables.append({"type": "enemy", "data": enemy, "sort_key": enemy["pos"][0] + enemy["pos"][1] + 0.5})

        renderables.sort(key=lambda r: r["sort_key"])

        # 3. Draw sorted objects
        for item in renderables:
            if item["type"] == "block":
                self._draw_iso_cube(item["pos"], self.COLOR_BLOCK, self.COLOR_BLOCK_TOP)
            elif item["type"] == "base":
                self._draw_iso_cube(item["pos"], self.COLOR_BASE, self.COLOR_BASE_TOP, height_mod=1.5)
            elif item["type"] == "enemy":
                self._draw_enemy(item["data"])
        
        # 4. Draw cursor
        self._draw_cursor()
        
        # 5. Draw projectiles and particles (on top)
        for proj in self.projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 20.0))))
            color = p["color"] + (alpha,)
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            temp_surf.fill(color)
            self.screen.blit(temp_surf, (int(p["pos"][0]), int(p["pos"][1])))

    def _draw_iso_cube(self, pos, side_color, top_color, height_mod=1.0):
        x, y = pos
        height = self.TILE_HEIGHT_HALF * 2 * height_mod
        
        base_p_top = self._cart_to_iso(x, y)
        p_top = (base_p_top[0], base_p_top[1] - height)
        p_right = (p_top[0] + self.TILE_WIDTH_HALF, p_top[1] + self.TILE_HEIGHT_HALF)
        p_left = (p_top[0] - self.TILE_WIDTH_HALF, p_top[1] + self.TILE_HEIGHT_HALF)
        p_top_right = (p_top[0] + self.TILE_WIDTH_HALF, p_top[1] - self.TILE_HEIGHT_HALF)

        p_bottom = self._cart_to_iso(x + 1, y + 1)
        p_bottom_right = self._cart_to_iso(x + 1, y)
        p_bottom_left = self._cart_to_iso(x, y + 1)
        
        # Top face
        pygame.gfxdraw.aapolygon(self.screen, [p_top, p_right, p_bottom_right, p_left], top_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_top, p_right, p_bottom_right, p_left], top_color)

        darker_side = tuple(max(0, c-25) for c in side_color)
        # Left face
        pygame.gfxdraw.aapolygon(self.screen, [p_left, p_bottom_left, p_bottom, p_bottom_right], darker_side)
        pygame.gfxdraw.filled_polygon(self.screen, [p_left, p_bottom_left, p_bottom, p_bottom_right], darker_side)
        
        # Right face
        pygame.gfxdraw.aapolygon(self.screen, [p_right, p_bottom_right, p_bottom, p_bottom_left], side_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_right, p_bottom_right, p_bottom, p_bottom_left], side_color)

    def _draw_enemy(self, enemy):
        bob_offset = math.sin(enemy["bob"]) * 2
        iso_pos = self._cart_to_iso(enemy["pos"][0], enemy["pos"][1], offset_y=-10 + bob_offset)
        x, y = int(iso_pos[0]), int(iso_pos[1])

        # Shadow
        shadow_pos = self._cart_to_iso(enemy["pos"][0], enemy["pos"][1], offset_y=2)
        pygame.gfxdraw.filled_ellipse(self.screen, int(shadow_pos[0]), int(shadow_pos[1]), 8, 4, self.COLOR_ENEMY_SHADOW)
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, x, y, 7, self.COLOR_ENEMY)

        # Health bar
        if enemy["health"] < enemy["max_health"]:
            bar_width = 20
            health_pct = enemy["health"] / enemy["max_health"]
            health_bar_rect = pygame.Rect(x - bar_width // 2, y - 20, bar_width, 4)
            fill_rect = pygame.Rect(x - bar_width // 2, y - 20, int(bar_width * health_pct), 4)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, health_bar_rect)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fill_rect)

    def _draw_cursor(self):
        x, y = self.cursor_pos
        p1 = self._cart_to_iso(x, y)
        p2 = self._cart_to_iso(x + 1, y)
        p3 = self._cart_to_iso(x + 1, y + 1)
        p4 = self._cart_to_iso(x, y + 1)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_CURSOR)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], self.COLOR_CURSOR)

    def _render_ui(self):
        # Base Health
        health_text = f"Base Health: {self.base_health}/{self.max_base_health}"
        text_surf = self.font_small.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Wave Counter
        wave_text = f"Wave: {self.wave}/{self.WIN_WAVE}"
        text_surf = self.font_large.render(wave_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(centerx=self.WIDTH / 2, top=10)
        self.screen.blit(text_surf, text_rect)
        
        # Score
        score_text = f"Score: {int(self.score)}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(right=self.WIDTH - 10, top=10)
        self.screen.blit(text_surf, text_rect)
        
        if self.wave_cooldown > 0 and self.wave <= self.WIN_WAVE:
            cooldown_sec = math.ceil(self.wave_cooldown / 30)
            wave_start_text = f"Next wave in {cooldown_sec}..."
            text_surf = self.font_small.render(wave_start_text, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(centerx=self.WIDTH/2, top=50)
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies) + len(self.enemies_to_spawn_this_wave)
        }

    def _cart_to_iso(self, x, y, offset_y=0):
        iso_x = self.ISO_OFFSET_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_HEIGHT_HALF + offset_y
        return iso_x, iso_y

    def _dist_sq(self, p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def validate_implementation(self):
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

    def close(self):
        pygame.quit()