
# Generated: 2025-08-27T16:09:41.595915
# Source Brief: brief_01134.md
# Brief Index: 1134

        
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


# Helper class for a 2D Vector
class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vec2(0, 0)
        return Vec2(self.x / mag, self.y / mag)

# Game Entity Classes
class Enemy:
    def __init__(self, path, health, speed, size=8, value=10):
        self.path = path
        self.path_index = 0
        self.pos = Vec2(path[0][0], path[0][1])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.size = size
        self.value = value

    def move(self):
        if self.path_index >= len(self.path) - 1:
            return True # Reached the end

        target_pos = Vec2(self.path[self.path_index + 1][0], self.path[self.path_index + 1][1])
        direction = (target_pos - self.pos).normalize()
        self.pos += direction * self.speed

        if (target_pos - self.pos).magnitude() < self.speed:
            self.pos = target_pos
            self.path_index += 1
        
        return False # Still on path

class Tower:
    def __init__(self, grid_pos, tower_type_data, cell_size):
        self.grid_pos = grid_pos
        self.pos = Vec2(grid_pos[0] * cell_size + cell_size / 2, grid_pos[1] * cell_size + cell_size / 2)
        self.type_data = tower_type_data
        self.range = tower_type_data['range']
        self.damage = tower_type_data['damage']
        self.fire_rate = tower_type_data['fire_rate']
        self.color = tower_type_data['color']
        self.cooldown = 0

    def update(self, enemies, projectiles):
        if self.cooldown > 0:
            self.cooldown -= 1
            return

        target = self._find_target(enemies)
        if target and self.cooldown <= 0:
            # sfx: tower_shoot
            projectiles.append(Projectile(self.pos, target, self.damage, self.type_data['projectile_speed'], self.type_data['projectile_color']))
            self.cooldown = self.fire_rate

    def _find_target(self, enemies):
        closest_enemy = None
        min_dist = float('inf')
        for enemy in enemies:
            dist = (enemy.pos - self.pos).magnitude()
            if dist <= self.range and dist < min_dist:
                min_dist = dist
                closest_enemy = enemy
        return closest_enemy

class Projectile:
    def __init__(self, start_pos, target, damage, speed, color):
        self.pos = Vec2(start_pos.x, start_pos.y)
        self.target = target
        self.damage = damage
        self.speed = speed
        self.color = color
        self.is_dead = False

    def move(self):
        if self.is_dead or not self.target or self.target.health <= 0:
            self.is_dead = True
            return False

        direction = (self.target.pos - self.pos).normalize()
        self.pos += direction * self.speed

        if (self.target.pos - self.pos).magnitude() < self.target.size:
            self.is_dead = True
            return True # Hit
        return False

class Particle:
    def __init__(self, x, y, color, life, size, velocity):
        self.pos = Vec2(x, y)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.velocity = velocity

    def update(self):
        self.pos += self.velocity
        self.life -= 1
        return self.life <= 0

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = "Controls: Use arrow keys to move the placement cursor. Press 'Shift' to cycle tower types. Press 'Space' to build a tower."
    game_description = "A minimalist tower defense game. Place towers to defend your base from waves of enemies. Survive all waves to win."
    auto_advance = True

    WIDTH, HEIGHT = 640, 400
    CELL_SIZE = 20
    GRID_WIDTH, GRID_HEIGHT = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE
    MAX_STEPS = 4500
    NUM_WAVES = 5
    
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 40)
    COLOR_PATH = (25, 25, 35)
    COLOR_BASE = (0, 150, 50)
    COLOR_SPAWN = (150, 0, 50)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_INVALID = (255, 0, 0, 100)
    
    TOWER_DATA = [
        {"name": "Gatling", "cost": 25, "range": 80, "damage": 0.5, "fire_rate": 8, "color": (0, 180, 220), "projectile_speed": 10, "projectile_color": (173, 216, 230)},
        {"name": "Cannon", "cost": 75, "range": 150, "damage": 5, "fire_rate": 45, "color": (255, 165, 0), "projectile_speed": 7, "projectile_color": (255, 200, 100)},
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 20)
        self.font_l = pygame.font.SysFont("monospace", 40, bold=True)
        
        self.enemy_path = self._define_path()
        self.reset()
    
    def _define_path(self):
        path = []
        path.append((-self.CELL_SIZE, 3 * self.CELL_SIZE))
        path.append((5 * self.CELL_SIZE, 3 * self.CELL_SIZE))
        path.append((5 * self.CELL_SIZE, 12 * self.CELL_SIZE))
        path.append((15 * self.CELL_SIZE, 12 * self.CELL_SIZE))
        path.append((15 * self.CELL_SIZE, 5 * self.CELL_SIZE))
        path.append((25 * self.CELL_SIZE, 5 * self.CELL_SIZE))
        path.append((25 * self.CELL_SIZE, 15 * self.CELL_SIZE))
        path.append((self.WIDTH + self.CELL_SIZE, 15 * self.CELL_SIZE))
        return path

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.total_reward = 0
        self.game_over = False
        self.game_result = 0
        
        self.base_health = 10
        self.resources = 80
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_type = 0
        
        self.prev_space_state = 0
        self.prev_shift_state = 0
        
        self.current_wave = 0
        self.wave_cooldown = 120
        self.enemies_to_spawn = []
        
        self.occupied_cells = set()
        for i in range(len(self.enemy_path) - 1):
            p1 = self.enemy_path[i]
            p2 = self.enemy_path[i+1]
            x1, y1 = p1[0] // self.CELL_SIZE, p1[1] // self.CELL_SIZE
            x2, y2 = p2[0] // self.CELL_SIZE, p2[1] // self.CELL_SIZE
            for x in range(min(x1, x2) -1, max(x1, x2) + 2):
                for y in range(min(y1, y2) -1, max(y1, y2) + 2):
                    if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                        self.occupied_cells.add((x, y))

        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.NUM_WAVES: return

        # sfx: wave_start
        num_enemies = 5 + (self.current_wave - 1) * 3
        enemy_health = 10 + (self.current_wave - 1) * 5
        enemy_speed = 1.0 + (self.current_wave - 1) * 0.1
        
        self.enemies_to_spawn = []
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({"health": enemy_health, "speed": enemy_speed, "delay": random.randint(30, 90)})

    def step(self, action):
        reward = -0.01 if self.base_health > 0 else 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        
        self._update_wave_spawning()
        reward += self._update_projectiles()
        self._update_towers()
        
        if self._update_enemies():
            self.base_health -= 1
            # sfx: base_damage
            self._create_explosion(self.enemy_path[-1][0] - self.CELL_SIZE, self.enemy_path[-1][1], self.COLOR_ENEMY, 30)
            if self.base_health <= 0:
                self.game_over = True
                self.game_result = -1
                reward = -100

        self._update_particles()
        
        terminated = self.game_over
        if not terminated and self.current_wave > self.NUM_WAVES and not self.enemies and not self.enemies_to_spawn:
            self.game_over = True
            self.game_result = 1
            reward = 100
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over: reward = -100 # Penalize for timeout
            
        self.steps += 1
        self.total_reward += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        if shift_held and not self.prev_shift_state:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_DATA)
            # sfx: ui_cycle
        self.prev_shift_state = shift_held
        
        if space_held and not self.prev_space_state:
            tower_data = self.TOWER_DATA[self.selected_tower_type]
            can_place = self.resources >= tower_data['cost'] and tuple(self.cursor_pos) not in self.occupied_cells
            if can_place:
                # sfx: place_tower
                self.resources -= tower_data['cost']
                self.towers.append(Tower(tuple(self.cursor_pos), tower_data, self.CELL_SIZE))
                self.occupied_cells.add(tuple(self.cursor_pos))
            else:
                # sfx: place_fail
                pass
        self.prev_space_state = space_held

    def _update_wave_spawning(self):
        if self.wave_cooldown > 0:
            self.wave_cooldown -= 1
            if self.wave_cooldown == 0:
                self._start_next_wave()
        elif not self.enemies and not self.enemies_to_spawn and self.current_wave < self.NUM_WAVES:
            self.wave_cooldown = 180
        
        if self.enemies_to_spawn and self.enemies_to_spawn[0]['delay'] > 0:
            self.enemies_to_spawn[0]['delay'] -= 1
        elif self.enemies_to_spawn:
            spawn_data = self.enemies_to_spawn.pop(0)
            self.enemies.append(Enemy(self.enemy_path, spawn_data['health'], spawn_data['speed']))

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            if p.move():
                # sfx: hit_enemy
                p.target.health -= p.damage
                reward += 0.1
                self._create_hit_effect(p.pos.x, p.pos.y, p.color)
                if p.target.health <= 0:
                    # sfx: kill_enemy
                    reward += 1.0
                    self.resources += p.target.value
                    self._create_explosion(p.target.pos.x, p.target.pos.y, self.COLOR_ENEMY, 20)
                    self.enemies.remove(p.target)
                self.projectiles.remove(p)
            elif p.is_dead:
                self.projectiles.remove(p)
        return reward

    def _update_towers(self):
        for tower in self.towers:
            tower.update(self.enemies, self.projectiles)
    
    def _update_enemies(self):
        reached_base = False
        for enemy in self.enemies[:]:
            if enemy.move():
                reached_base = True
                self.enemies.remove(enemy)
        return reached_base

    def _update_particles(self):
        for p in self.particles[:]:
            if p.update():
                self.particles.remove(p)

    def _create_hit_effect(self, x, y, color):
        for _ in range(3):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5)
            vel = Vec2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(x, y, color, 10, random.randint(1, 3), vel))

    def _create_explosion(self, x, y, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = Vec2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append(Particle(x, y, color, random.randint(15, 30), random.randint(2, 4), vel))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.total_reward, "steps": self.steps}

    def _render_game(self):
        for x in range(0, self.WIDTH, self.CELL_SIZE): pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE): pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        if len(self.enemy_path) > 1: pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.enemy_path, self.CELL_SIZE)
        
        pygame.draw.circle(self.screen, self.COLOR_SPAWN, self.enemy_path[0], self.CELL_SIZE // 2)
        base_rect = pygame.Rect(self.enemy_path[-1][0] - self.CELL_SIZE, self.enemy_path[-1][1] - self.CELL_SIZE // 2, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        
        self._render_cursor_and_range()

        for tower in self.towers:
            rect = pygame.Rect(tower.grid_pos[0] * self.CELL_SIZE, tower.grid_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, tower.color, rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in tower.color), rect, width=2, border_radius=3)

        for enemy in self.enemies:
            pos = (int(enemy.pos.x), int(enemy.pos.y))
            size = enemy.size
            points = [(pos[0], pos[1] - size), (pos[0] - size / 1.5, pos[1] + size / 2), (pos[0] + size / 1.5, pos[1] + size / 2)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ENEMY)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ENEMY)
            health_pct = max(0, enemy.health / enemy.max_health)
            bar_w = size * 2
            pygame.draw.rect(self.screen, (100,0,0), (pos[0] - bar_w/2, pos[1] - size - 8, bar_w, 4))
            pygame.draw.rect(self.screen, (0,200,0), (pos[0] - bar_w/2, pos[1] - size - 8, bar_w * health_pct, 4))
            
        for p in self.projectiles: pygame.draw.circle(self.screen, p.color, (int(p.pos.x), int(p.pos.y)), 3)

        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            temp_surf = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p.size, p.size), p.size)
            self.screen.blit(temp_surf, (int(p.pos.x - p.size), int(p.pos.y - p.size)))

    def _render_cursor_and_range(self):
        cx, cy = self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE
        tower_data = self.TOWER_DATA[self.selected_tower_type]
        can_afford, is_occupied = self.resources >= tower_data['cost'], tuple(self.cursor_pos) in self.occupied_cells
        
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        range_color = (*tower_data['color'], 50) if can_afford and not is_occupied else (100, 100, 100, 50)
        pygame.draw.circle(s, range_color, (cx + self.CELL_SIZE//2, cy + self.CELL_SIZE//2), tower_data['range'])
        self.screen.blit(s, (0, 0))

        cursor_color = self.COLOR_CURSOR if can_afford and not is_occupied else (150, 150, 150)
        pygame.draw.rect(self.screen, cursor_color, (cx, cy, self.CELL_SIZE, self.CELL_SIZE), 2)

        if is_occupied:
            pygame.draw.line(self.screen, (255,0,0), (cx+2, cy+2), (cx+self.CELL_SIZE-2, cy+self.CELL_SIZE-2), 2)
            pygame.draw.line(self.screen, (255,0,0), (cx+self.CELL_SIZE-2, cy+2), (cx+2, cy+self.CELL_SIZE-2), 2)
        
        if self.prev_space_state and (not can_afford or is_occupied):
            s_flash = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s_flash.fill(self.COLOR_INVALID)
            self.screen.blit(s_flash, (cx, cy))

    def _render_ui(self):
        wave_text = f"Wave: {min(self.current_wave, self.NUM_WAVES)}/{self.NUM_WAVES}"
        if self.wave_cooldown > 0 and self.current_wave < self.NUM_WAVES: wave_text = f"Next wave in: {self.wave_cooldown // 30 + 1}"
        self.screen.blit(self.font_m.render(wave_text, True, self.COLOR_TEXT), (10, 10))

        enemy_count = len(self.enemies) + len(self.enemies_to_spawn)
        self.screen.blit(self.font_m.render(f"Enemies: {enemy_count}", True, self.COLOR_TEXT), (self.WIDTH - 150, 10))

        self.screen.blit(self.font_m.render(f"$ {self.resources}", True, (255, 223, 0)), (10, self.HEIGHT - 60))
        self.screen.blit(self.font_m.render(f"HP: {self.base_health}", True, self.COLOR_BASE), (10, self.HEIGHT - 35))
        
        tower_data = self.TOWER_DATA[self.selected_tower_type]
        name_surf = self.font_m.render(f"Build: {tower_data['name']}", True, self.COLOR_TEXT)
        cost_color = (255, 223, 0) if self.resources >= tower_data['cost'] else self.COLOR_ENEMY
        cost_surf = self.font_m.render(f"Cost: ${tower_data['cost']}", True, cost_color)
        self.screen.blit(name_surf, (self.WIDTH - name_surf.get_width() - 10, self.HEIGHT - 60))
        self.screen.blit(cost_surf, (self.WIDTH - cost_surf.get_width() - 10, self.HEIGHT - 35))
        
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (0, 0))
            end_text, end_color = ("VICTORY", (0, 255, 128)) if self.game_result == 1 else ("GAME OVER", (255, 50, 50))
            text_surf = self.font_l.render(end_text, True, end_color)
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

if __name__ == '__main__':
    env = GameEnv()
    
    obs, info = env.reset()
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    while running:
        action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over. Final Score: {info['score']:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        env.clock.tick(30)
        
    pygame.quit()