
# Generated: 2025-08-27T12:44:40.713614
# Source Brief: brief_00142.md
# Brief Index: 142

        
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


# Helper classes for game entities to keep the main environment class clean.

class Tower:
    """Represents a single defensive tower."""
    def __init__(self, grid_pos, tower_type, tower_stats):
        self.grid_pos = grid_pos
        self.type = tower_type
        self.stats = tower_stats[self.type]
        self.range = self.stats["range"]
        self.damage = self.stats["damage"]
        self.fire_rate = self.stats["fire_rate"]
        self.cooldown = 0
        self.target = None

    def find_target(self, enemies):
        """Finds the best enemy to target within range."""
        # If current target is out of range or dead, find a new one
        if self.target and (self.target.health <= 0 or self._distance_to(self.target) > self.range):
            self.target = None

        if not self.target:
            in_range_enemies = [e for e in enemies if self._distance_to(e) <= self.range]
            if in_range_enemies:
                # Target the enemy that is furthest along the path
                self.target = max(in_range_enemies, key=lambda e: e.path_progress)

    def _distance_to(self, enemy):
        """Calculates grid distance to an enemy."""
        return math.hypot(self.grid_pos[0] - enemy.grid_pos[0], self.grid_pos[1] - enemy.grid_pos[1])

    def update(self, enemies):
        """Updates tower state, finds target, and fires if ready."""
        self.cooldown = max(0, self.cooldown - 1)
        self.find_target(enemies)
        if self.target and self.cooldown == 0:
            # Game logic assumes 60fps for smooth calculations, env steps at 30fps
            self.cooldown = 60 / self.fire_rate
            # sfx: shoot.wav
            return Projectile(self.grid_pos, self.target, self.damage, self.stats["proj_speed"], self.stats["proj_color"])
        return None

class Enemy:
    """Represents a single enemy unit moving along the path."""
    def __init__(self, wave_health, wave_speed, path):
        self.path = path
        self.path_index = 0
        self.progress = 0.0
        self.max_health = wave_health
        self.health = wave_health
        self.speed = wave_speed / 60.0  # Per-frame speed at 60fps logic rate
        self.grid_pos = self.path[0]
        self.path_progress = 0
        self.is_dead = False
        self.reached_end = False

    def move(self):
        """Moves the enemy along its predefined path."""
        if self.path_index >= len(self.path) - 1:
            self.reached_end = True
            return

        self.progress += self.speed
        self.path_progress += self.speed

        if self.progress >= 1.0:
            self.progress -= 1.0
            self.path_index += 1
            if self.path_index >= len(self.path) - 1:
                self.reached_end = True
                self.grid_pos = self.path[-1]
                return

        start_node = self.path[self.path_index]
        end_node = self.path[self.path_index + 1]
        self.grid_pos = (
            start_node[0] + (end_node[0] - start_node[0]) * self.progress,
            start_node[1] + (end_node[1] - start_node[1]) * self.progress,
        )

    def take_damage(self, amount):
        """Reduces enemy health and checks for death."""
        self.health -= amount
        if self.health <= 0:
            self.is_dead = True
        return self.is_dead

class Projectile:
    """Represents a projectile fired from a tower."""
    def __init__(self, start_grid_pos, target, damage, speed, color):
        self.grid_pos = start_grid_pos
        self.target = target
        self.damage = damage
        self.speed = speed / 60.0 # Per-frame speed at 60fps logic rate
        self.color = color
        self.to_be_removed = False

    def move(self):
        """Moves the projectile towards its target."""
        if self.target.health <= 0:
            self.to_be_removed = True
            return

        direction_x = self.target.grid_pos[0] - self.grid_pos[0]
        direction_y = self.target.grid_pos[1] - self.grid_pos[1]
        distance = math.hypot(direction_x, direction_y)

        if distance < self.speed:
            self.grid_pos = self.target.grid_pos
            self.to_be_removed = True
        else:
            self.grid_pos = (
                self.grid_pos[0] + (direction_x / distance) * self.speed,
                self.grid_pos[1] + (direction_y / distance) * self.speed,
            )

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, pos, color, size, life, velocity):
        self.pos = list(pos)
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.velocity = velocity

    def update(self):
        """Updates particle position and lifetime."""
        self.pos[0] += self.velocity[0]
        self.pos[1] += self.velocity[1]
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        """Draws the particle with alpha blending."""
        alpha = int(255 * (self.life / self.max_life))
        # Pygame doesn't handle alpha in its main draw functions well, need to create a temp surface
        temp_surface = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surface, (*self.color, alpha), (self.size, self.size), self.size)
        surface.blit(temp_surface, (int(self.pos[0] - self.size), int(self.pos[1] - self.size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Arrows to move cursor, Space to place tower, Shift to cycle tower type."
    game_description = "Defend your base from waves of enemies by strategically placing towers in this isometric tower defense game."
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game constants
        self.max_steps = 30 * 180 # 3 minutes at 30 fps
        self.total_waves = 10
        self.base_start_health = 100
        self.start_resources = 150
        
        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_PLACEABLE = (50, 65, 50)
        self.COLOR_BASE = (50, 100, 200)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CURSOR_VALID = (100, 255, 100)
        self.COLOR_CURSOR_INVALID = (255, 100, 100)

        # Isometric grid setup
        self.grid_w, self.grid_h = 20, 14
        self.tile_w, self.tile_h = 32, 16
        self.origin_x, self.origin_y = self.width // 2, 80

        # Game assets (defined in code)
        self.path = [(0, 6), (4, 6), (4, 2), (10, 2), (10, 10), (16, 10), (16, 5), (19, 5)]
        self.placeable_tiles = self._generate_placeable_tiles()
        self.tower_stats = [
            {"name": "Gatling", "cost": 50, "range": 3.5, "damage": 1, "fire_rate": 6, "proj_speed": 10, "color": (0, 200, 0), "proj_color": (255, 255, 0)},
            {"name": "Cannon", "cost": 120, "range": 5.0, "damage": 8, "fire_rate": 1, "proj_speed": 6, "color": (0, 150, 150), "proj_color": (255, 150, 0)},
        ]

        self.reset()
        self.validate_implementation()
    
    def _generate_placeable_tiles(self):
        tiles = set()
        path_set = set()
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            if p1[0] == p2[0]:
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                    path_set.add((p1[0], y))
            else:
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
                    path_set.add((x, p1[1]))
        
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if (x,y) not in path_set:
                    tiles.add((x,y))
        return list(tiles)

    def _iso_to_screen(self, x, y):
        screen_x = self.origin_x + (x - y) * self.tile_w / 2
        screen_y = self.origin_y + (x + y) * self.tile_h / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_poly(self, surface, x, y, color, height=0):
        p1, p2, p3, p4 = self._iso_to_screen(x, y), self._iso_to_screen(x + 1, y), self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x, y + 1)
        points = [(p[0], p[1] - height) for p in [p1, p2, p3, p4]]
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.steps, self.score, self.game_over, self.game_won = 0, 0, False, False
        self.base_health, self.resources = self.base_start_health, self.start_resources
        self.wave_number, self.wave_in_progress, self.wave_finished_timer, self.wave_spawn_timer, self.enemies_to_spawn = 0, False, 0, 0, 0
        self.enemies, self.towers, self.projectiles, self.particles = [], [], [], []
        self.cursor_pos, self.selected_tower_type = [self.grid_w // 2, self.grid_h // 2], 0
        self.last_shift_press, self.last_space_press = False, False
        self._start_next_wave()
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.total_waves:
            self.game_won = True
            self.game_over = True
            return
        self.wave_in_progress = True
        self.enemies_to_spawn = 5 + self.wave_number * 2
        self.wave_spawn_timer = 0
        self.wave_enemy_health = 5 + self.wave_number
        self.wave_enemy_speed = 0.5 + self.wave_number * 0.05
        
    def step(self, action):
        if self.auto_advance: self.clock.tick(30)
        reward = 0
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            self._handle_input(movement, shift_held)
            if space_held and not self.last_space_press: self._place_tower()
            self.last_space_press = space_held
            
            # Logic update is run twice per frame for 60fps-like smoothness
            for _ in range(2):
                self._update_waves()
                self._update_towers()
                hit_reward = self._update_projectiles()
                leak_reward, kill_reward = self._update_enemies()
                self._update_particles()
                reward += (hit_reward + leak_reward + kill_reward) / 2.0

            if self.wave_in_progress and self.enemies_to_spawn == 0 and not self.enemies:
                self.wave_in_progress = False
                self.wave_finished_timer = 120 # 2 second pause at 60 logic ticks
                reward += 50
        
        self.steps += 1
        terminated = self.game_over or self.steps >= self.max_steps
        if self.base_health <= 0 and not self.game_over:
            self.base_health = 0
            reward += -10
            terminated = True
        if self.game_won and not self.game_over:
            reward += 100
            terminated = True
        self.game_over = terminated
        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, shift_held):
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.grid_h - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.grid_w - 1, self.cursor_pos[0] + 1)
        if shift_held and not self.last_shift_press: self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_stats)
        self.last_shift_press = shift_held

    def _place_tower(self):
        pos, stats = tuple(self.cursor_pos), self.tower_stats[self.selected_tower_type]
        if pos in self.placeable_tiles and not any(t.grid_pos == pos for t in self.towers) and self.resources >= stats["cost"]:
            self.resources -= stats["cost"]
            self.towers.append(Tower(pos, self.selected_tower_type, self.tower_stats))
            # sfx: tower_place.wav
        # else: sfx: error.wav

    def _update_waves(self):
        if not self.wave_in_progress:
            if self.wave_finished_timer > 0 and not self.game_over:
                self.wave_finished_timer -= 1
                if self.wave_finished_timer == 0: self._start_next_wave()
            return
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.enemies_to_spawn > 0:
            self.enemies.append(Enemy(self.wave_enemy_health, self.wave_enemy_speed, self.path))
            self.enemies_to_spawn -= 1
            self.wave_spawn_timer = 60 # Spawn delay

    def _update_towers(self):
        for tower in self.towers:
            if projectile := tower.update(self.enemies): self.projectiles.append(projectile)

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p.move()
            if p.to_be_removed:
                if math.hypot(p.grid_pos[0] - p.target.grid_pos[0], p.grid_pos[1] - p.target.grid_pos[1]) < 0.5:
                    if p.target.health > 0:
                        p.target.take_damage(p.damage)
                        reward += 0.1 # sfx: hit.wav
                        self._create_particles(self._iso_to_screen(*p.target.grid_pos), (255, 200, 100), 5)
                self.projectiles.remove(p)
        return reward
        
    def _update_enemies(self):
        leak_reward, kill_reward = 0, 0
        for e in self.enemies[:]:
            e.move()
            if e.is_dead:
                self.resources += self.wave_enemy_health
                kill_reward += 1 # sfx: enemy_die.wav
                self.enemies.remove(e)
            elif e.reached_end:
                self.base_health -= 5
                leak_reward -= 5 # sfx: base_damage.wav
                self.enemies.remove(e)
        return leak_reward, kill_reward

    def _update_particles(self):
        for p in self.particles[:]:
            if p.update(): self.particles.remove(p)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.random() * 2 + 1
            life = self.np_random.integers(10, 20)
            self.particles.append(Particle(pos, color, size, life, velocity))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for x, y in self.placeable_tiles: self._draw_iso_poly(self.screen, x, y, self.COLOR_PLACEABLE)
        for i in range(len(self.path) - 1):
            p1, p2 = self.path[i], self.path[i+1]
            if p1[0] == p2[0]:
                for y in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1): self._draw_iso_poly(self.screen, p1[0], y, self.COLOR_PATH)
            else:
                for x in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1): self._draw_iso_poly(self.screen, x, p1[1], self.COLOR_PATH)
        self._draw_iso_poly(self.screen, self.path[-1][0], self.path[-1][1], self.COLOR_BASE, height=5)
        
        drawables = sorted(self.towers + self.enemies, key=lambda obj: self._iso_to_screen(*obj.grid_pos)[1])
        for obj in drawables:
            sx, sy = self._iso_to_screen(*obj.grid_pos)
            if isinstance(obj, Tower):
                pygame.gfxdraw.filled_circle(self.screen, sx, sy - 8, 8, obj.stats["color"])
                pygame.gfxdraw.aacircle(self.screen, sx, sy - 8, 8, obj.stats["color"])
            elif isinstance(obj, Enemy):
                size = 6
                pygame.gfxdraw.filled_circle(self.screen, sx, sy - size, size, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, sx, sy - size, size, self.COLOR_ENEMY)
                pygame.draw.rect(self.screen, (255,0,0), (sx-size, sy-size*2-2, size*2, 3))
                pygame.draw.rect(self.screen, (0,255,0), (sx-size, sy-size*2-2, int(size*2*(obj.health/obj.max_health)), 3))

        for p in self.projectiles:
            sx, sy = self._iso_to_screen(*p.grid_pos)
            pygame.draw.circle(self.screen, p.color, (sx, sy - 4), 3)
        for p in self.particles: p.draw(self.screen)
        self._render_cursor()

    def _render_cursor(self):
        pos, stats = tuple(self.cursor_pos), self.tower_stats[self.selected_tower_type]
        valid = pos in self.placeable_tiles and not any(t.grid_pos == pos for t in self.towers) and self.resources >= stats["cost"]
        color = self.COLOR_CURSOR_VALID if valid else self.COLOR_CURSOR_INVALID
        sx, sy = self._iso_to_screen(*pos)
        
        temp_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._draw_iso_poly(temp_surface, pos[0], pos[1], (*color, 100))
        range_px = int(stats["range"] * self.tile_w / 2 * 1.5) # Scale range visually
        pygame.gfxdraw.aacircle(temp_surface, sx, sy, range_px, (*color, 150))
        self.screen.blit(temp_surface, (0, 0))

    def _render_ui(self):
        self.screen.blit(self.font_m.render(f"$: {self.resources}", True, self.COLOR_TEXT), (10, 10))
        wave_text = self.font_m.render(f"Wave: {self.wave_number}/{self.total_waves}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.width // 2 - wave_text.get_width() // 2, 10))
        
        health_pct = max(0, self.base_health / self.base_start_health)
        pygame.draw.rect(self.screen, (200,0,0), (self.width - 160, 10, 150, 20))
        pygame.draw.rect(self.screen, (0,200,0), (self.width - 160, 10, int(150 * health_pct), 20))
        self.screen.blit(self.font_s.render(f"Base: {self.base_health}%", True, self.COLOR_TEXT), (self.width - 155, 13))

        stats = self.tower_stats[self.selected_tower_type]
        self.screen.blit(self.font_s.render(f"Selected: {stats['name']} (Cost: {stats['cost']})", True, self.COLOR_TEXT), (10, self.height - 25))
        
        if self.game_over:
            msg, color = ("YOU WIN!", (50, 255, 50)) if self.game_won else ("GAME OVER", (255, 50, 50))
            end_text = self.font_l.render(msg, True, color)
            self.screen.blit(end_text, (self.width//2 - end_text.get_width()//2, self.height//2 - end_text.get_height()//2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "resources": self.resources, "base_health": self.base_health, "enemies_left": len(self.enemies) + self.enemies_to_spawn}

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3) and test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (self.height, self.width, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and trunc is False and isinstance(info, dict)
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Isometric Tower Defense")
    clock = pygame.time.Clock()
    obs, info = env.reset()
    terminated = False
    
    while not terminated:
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30)
    env.close()