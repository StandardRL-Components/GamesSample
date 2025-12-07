
# Generated: 2025-08-28T04:46:18.470136
# Source Brief: brief_05354.md
# Brief Index: 5354

        
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

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place selected tower. Shift to cycle tower type."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing procedurally generated towers in this top-down tower defense game."
    )

    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_SIZE = 40
    GRID_W, GRID_H = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE
    FPS = 30
    MAX_STEPS = 30 * 120 # 2 minutes at 30fps
    MAX_WAVES = 15

    # --- Colors ---
    COLOR_BG = (15, 19, 23)
    COLOR_PATH = (40, 45, 50)
    COLOR_GRID = (30, 35, 40)
    COLOR_BASE = (0, 150, 136)
    COLOR_ENEMY = (244, 67, 54)
    COLOR_TEXT = (230, 230, 230)
    COLOR_CURSOR = (255, 255, 255)
    
    TOWER_SPECS = {
        1: {"name": "Gatling", "cost": 75, "range": 80, "damage": 4, "fire_rate": 5, "color": (33, 150, 243), "shape": "triangle", "projectile_speed": 12},
        2: {"name": "Sniper", "cost": 125, "range": 200, "damage": 25, "fire_rate": 30, "color": (255, 235, 59), "shape": "diamond", "projectile_speed": 20},
        3: {"name": "Splash", "cost": 100, "range": 120, "damage": 8, "fire_rate": 20, "color": (156, 39, 176), "shape": "hexagon", "projectile_speed": 8, "splash_radius": 30},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.path_waypoints = []
        self.path_rects = []
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.terminal_reward = 0

        self.base_health = 100
        self.gold = 150
        self.current_wave = 0
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 1
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.enemies_to_spawn = []

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._create_path()
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(self.FPS)
        self.steps += 1
        
        step_reward = -0.01  # Time penalty

        # --- Handle Input ---
        self._handle_input(action)
        
        # --- Update Game State ---
        killed_this_step, reached_base_this_step = self._update_enemies()
        self._update_towers()
        self._update_projectiles()
        self._update_particles()
        wave_cleared = self._update_waves()

        # --- Calculate Rewards ---
        step_reward += killed_this_step * 0.1
        self.score += killed_this_step * 10
        if wave_cleared:
            step_reward += 1.0
            self.score += 100
        
        # Base health is reduced inside _update_enemies
        if reached_base_this_step > 0:
            self.score -= reached_base_this_step * 50

        # --- Check Termination ---
        terminated = self._check_termination()
        reward = step_reward + self.terminal_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Place Tower (on key press) ---
        if space_held and not self.prev_space_held:
            self._place_tower()
        
        # --- Cycle Tower Type (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type % 3) + 1

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _create_path(self):
        self.path_waypoints = []
        self.path_rects = []
        path_points = [
            (-1, 3), (2, 3), (2, 7), (10, 7), (10, 2), (self.GRID_W, 2)
        ]
        for i in range(len(path_points) - 1):
            p1 = np.array(path_points[i]) * self.GRID_SIZE
            p2 = np.array(path_points[i+1]) * self.GRID_SIZE
            self.path_waypoints.append((p1, p2))
            
            # Create rects for placement collision
            start_x, start_y = path_points[i]
            end_x, end_y = path_points[i+1]
            if start_x == end_x: # Vertical
                for y in range(min(start_y, end_y), max(start_y, end_y) + 1):
                    self.path_rects.append(pygame.Rect(start_x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
            else: # Horizontal
                for x in range(min(start_x, end_x), max(start_x, end_x) + 1):
                    self.path_rects.append(pygame.Rect(x * self.GRID_SIZE, start_y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            return

        num_enemies = 5 + (self.current_wave - 1) * 2
        base_health = 10 + (self.current_wave - 1) * 2
        base_speed = 1.0 + (self.current_wave - 1) * 0.05
        
        self.enemies_to_spawn = []
        for i in range(num_enemies):
            spawn_delay = i * (self.FPS // 2) # Spawn every 0.5s
            self.enemies_to_spawn.append({
                "health": base_health * (1 + self.np_random.uniform(-0.1, 0.1)),
                "speed": base_speed * (1 + self.np_random.uniform(-0.1, 0.1)),
                "delay": spawn_delay
            })
        self.enemies_to_spawn.sort(key=lambda x: x['delay'])

    def _update_waves(self):
        # Spawn enemies
        if self.enemies_to_spawn:
            spawned_this_frame = []
            for i, enemy_data in enumerate(self.enemies_to_spawn):
                enemy_data['delay'] -= 1
                if enemy_data['delay'] <= 0:
                    self.enemies.append(Enemy(enemy_data['health'], enemy_data['speed'], self.path_waypoints))
                    spawned_this_frame.append(i)
            # Remove spawned enemies from list
            for i in sorted(spawned_this_frame, reverse=True):
                del self.enemies_to_spawn[i]
        
        # Check for wave clear
        if not self.enemies and not self.enemies_to_spawn and self.current_wave <= self.MAX_WAVES:
            if self.current_wave == self.MAX_WAVES:
                self.game_won = True
                return False # Don't start a new wave
            self.gold += 100 + self.current_wave * 10
            self._start_next_wave()
            return True
        return False

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.gold < spec["cost"]:
            return # Not enough gold
        
        cx, cy = self.cursor_pos
        pos = (cx * self.GRID_SIZE + self.GRID_SIZE // 2, cy * self.GRID_SIZE + self.GRID_SIZE // 2)
        
        # Check if placing on path
        cursor_rect = pygame.Rect(cx * self.GRID_SIZE, cy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        if any(cursor_rect.colliderect(r) for r in self.path_rects):
            return # Cannot place on path

        # Check if placing on another tower
        if any(t.grid_pos == self.cursor_pos for t in self.towers):
            return # Cannot place on another tower
        
        self.gold -= spec["cost"]
        self.towers.append(Tower(pos, self.cursor_pos, self.selected_tower_type))
        # SFX: place_tower.wav

    def _update_towers(self):
        for tower in self.towers:
            tower.cooldown -= 1
            if tower.cooldown <= 0:
                target = tower.find_target(self.enemies)
                if target:
                    # SFX: shoot.wav
                    self.projectiles.append(Projectile(tower, target, self.TOWER_SPECS))
                    tower.cooldown = self.TOWER_SPECS[tower.type]["fire_rate"]

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.move()
            if p.check_hit():
                # SFX: hit.wav
                spec = self.TOWER_SPECS[p.tower_type]
                if "splash_radius" in spec: # Splash damage
                    for enemy in self.enemies:
                        if math.hypot(enemy.pos[0] - p.pos[0], enemy.pos[1] - p.pos[1]) < spec["splash_radius"]:
                            enemy.take_damage(spec["damage"])
                    self.particles.append(Particle(p.pos, spec["color"], spec["splash_radius"], 0.3))
                else: # Single target
                    p.target.take_damage(spec["damage"])
                    self.particles.append(Particle(p.pos, spec["color"], 10, 0.2))

                if p in self.projectiles: self.projectiles.remove(p)

    def _update_enemies(self):
        killed_count = 0
        reached_base_count = 0
        for enemy in self.enemies[:]:
            if not enemy.is_alive:
                self.gold += 5
                killed_count += 1
                self.particles.append(Particle(enemy.pos, self.COLOR_ENEMY, 20, 0.5))
                self.enemies.remove(enemy)
                # SFX: enemy_die.wav
                continue
            
            if enemy.update(self.path_waypoints):
                self.base_health -= 10
                reached_base_count += 1
                self.enemies.remove(enemy)
                # SFX: base_damage.wav
        
        self.base_health = max(0, self.base_health)
        return killed_count, reached_base_count

    def _update_particles(self):
        for particle in self.particles[:]:
            if not particle.update():
                self.particles.remove(particle)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.base_health <= 0:
            self.game_over = True
            self.terminal_reward = -100
        elif self.game_won:
            self.game_over = True
            self.terminal_reward = 100
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.terminal_reward = -10 # Ran out of time
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_path()
        self._render_grid_and_cursor()
        self._render_base()
        for tower in self.towers: self._render_tower(tower)
        for enemy in self.enemies: self._render_enemy(enemy)
        for proj in self.projectiles: self._render_projectile(proj)
        for part in self.particles: self._render_particle(part)
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "wave": self.current_wave,
            "base_health": self.base_health,
        }

    # --- Rendering Methods ---
    def _render_path(self):
        for start, end in self.path_waypoints:
            pygame.draw.line(self.screen, self.COLOR_PATH, start, end, self.GRID_SIZE)

    def _render_grid_and_cursor(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.GRID_SIZE, cy * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        # Pulsating alpha for cursor
        alpha = 128 + int(127 * math.sin(self.steps * 0.2))
        
        # Draw tower range preview
        spec = self.TOWER_SPECS[self.selected_tower_type]
        range_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(range_surface, cursor_rect.centerx, cursor_rect.centery, spec["range"], (*spec["color"], 50))
        pygame.gfxdraw.aacircle(range_surface, cursor_rect.centerx, cursor_rect.centery, spec["range"], (*spec["color"], 100))
        self.screen.blit(range_surface, (0, 0))

        # Draw cursor rectangle
        cursor_surface = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        cursor_surface.fill((*self.COLOR_CURSOR, alpha))
        self.screen.blit(cursor_surface, cursor_rect.topleft)

    def _render_base(self):
        base_x = self.path_waypoints[-1][1][0]
        base_y = self.path_waypoints[-1][1][1] - self.GRID_SIZE // 2
        pygame.draw.rect(self.screen, self.COLOR_BASE, (base_x, base_y, self.GRID_SIZE, self.GRID_SIZE))

    def _render_tower(self, tower):
        spec = self.TOWER_SPECS[tower.type]
        pos = tower.pos
        color = spec["color"]
        size = self.GRID_SIZE // 2 - 4
        
        if spec["shape"] == "triangle":
            points = [(pos[0], pos[1] - size), (pos[0] - size, pos[1] + size//2), (pos[0] + size, pos[1] + size//2)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif spec["shape"] == "diamond":
            points = [(pos[0], pos[1] - size), (pos[0] + size, pos[1]), (pos[0], pos[1] + size), (pos[0] - size, pos[1])]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        elif spec["shape"] == "hexagon":
            points = []
            for i in range(6):
                angle = math.pi / 3 * i
                points.append((pos[0] + size * math.cos(angle), pos[1] + size * math.sin(angle)))
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_enemy(self, enemy):
        pos = (int(enemy.pos[0]), int(enemy.pos[1]))
        radius = 10
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
        
        # Health bar
        health_pct = enemy.health / enemy.max_health
        bar_width = 16
        bar_height = 3
        bar_x = pos[0] - bar_width // 2
        bar_y = pos[1] - radius - bar_height - 2
        pygame.draw.rect(self.screen, (80, 0, 0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 0), (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_projectile(self, proj):
        color = self.TOWER_SPECS[proj.tower_type]["color"]
        pygame.draw.rect(self.screen, color, (int(proj.pos[0])-2, int(proj.pos[1])-2, 4, 4))

    def _render_particle(self, particle):
        alpha = int(255 * (1 - particle.life / particle.lifespan))
        color = (*particle.color, alpha)
        surface = pygame.Surface((particle.radius*2, particle.radius*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(surface, particle.radius, particle.radius, int(particle.radius), color)
        self.screen.blit(surface, (int(particle.pos[0] - particle.radius), int(particle.pos[1] - particle.radius)))

    def _render_ui(self):
        # Health
        health_text = self.font_m.render(f"♥ {int(self.base_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        # Gold
        gold_text = self.font_m.render(f"♦ {self.gold}", True, self.COLOR_TEXT)
        self.screen.blit(gold_text, (self.WIDTH - gold_text.get_width() - 10, 10))
        # Wave
        wave_text = self.font_m.render(f"Wave {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH//2 - wave_text.get_width()//2, 10))

        # Selected Tower UI
        spec = self.TOWER_SPECS[self.selected_tower_type]
        tower_text = self.font_s.render(f"Build: {spec['name']} ({spec['cost']}♦)", True, spec["color"])
        self.screen.blit(tower_text, (10, self.HEIGHT - tower_text.get_height() - 10))

    def _render_game_over(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 180))
        self.screen.blit(s, (0, 0))
        
        message = "VICTORY!" if self.game_won else "GAME OVER"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        
        text = self.font_l.render(message, True, color)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(text, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# Helper classes for game entities
class Enemy:
    def __init__(self, health, speed, path_waypoints):
        self.max_health = health
        self.health = health
        self.speed = speed
        self.pos = np.array(path_waypoints[0][0], dtype=float)
        self.path_index = 0
        self.path_progress = 0.0
        self.is_alive = True

    def update(self, path_waypoints):
        if self.path_index >= len(path_waypoints):
            return True # Reached end

        p1, p2 = path_waypoints[self.path_index]
        path_vec = np.array(p2) - np.array(p1)
        path_len = np.linalg.norm(path_vec)
        
        if path_len > 0:
            self.path_progress += self.speed / path_len
            self.pos = np.array(p1) + self.path_progress * path_vec
        
        if self.path_progress >= 1.0:
            self.path_index += 1
            self.path_progress = 0.0
            if self.path_index < len(path_waypoints):
                self.pos = np.array(path_waypoints[self.path_index][0])
        
        return False

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.is_alive = False

class Tower:
    def __init__(self, pos, grid_pos, tower_type):
        self.pos = pos
        self.grid_pos = grid_pos
        self.type = tower_type
        self.cooldown = 0
    
    def find_target(self, enemies):
        spec = GameEnv.TOWER_SPECS[self.type]
        in_range = []
        for enemy in enemies:
            dist = math.hypot(self.pos[0] - enemy.pos[0], self.pos[1] - enemy.pos[1])
            if dist <= spec["range"]:
                in_range.append(enemy)
        
        if not in_range:
            return None
        
        # Target enemy furthest along the path
        return max(in_range, key=lambda e: (e.path_index, e.path_progress))

class Projectile:
    def __init__(self, tower, target, specs):
        self.tower_type = tower.type
        self.pos = np.array(tower.pos, dtype=float)
        self.target = target
        self.damage = specs[self.tower_type]["damage"]
        self.speed = specs[self.tower_type]["projectile_speed"]

    def move(self):
        if not self.target.is_alive: # If target is dead, keep going straight
            direction = self.last_direction
        else:
            direction = np.array(self.target.pos) - self.pos
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            self.last_direction = direction
        
        self.pos += direction * self.speed
    
    def check_hit(self):
        if not self.target.is_alive:
            return self.pos[0] < 0 or self.pos[0] > GameEnv.WIDTH or self.pos[1] < 0 or self.pos[1] > GameEnv.HEIGHT
        return math.hypot(self.pos[0] - self.target.pos[0], self.pos[1] - self.target.pos[1]) < 10

class Particle:
    def __init__(self, pos, color, max_radius, lifespan_s):
        self.pos = pos
        self.color = color
        self.max_radius = max_radius
        self.lifespan = lifespan_s * GameEnv.FPS
        self.life = self.lifespan
        self.radius = 0

    def update(self):
        self.life -= 1
        if self.life <= 0:
            return False
        
        progress = 1 - (self.life / self.lifespan)
        self.radius = self.max_radius * math.sin(progress * math.pi) # Swell and shrink effect
        return True

if __name__ == '__main__':
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tower Defense")
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, no-space, no-shift
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()
            
    pygame.quit()