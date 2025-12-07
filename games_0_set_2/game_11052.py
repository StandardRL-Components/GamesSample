import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:22:25.794059
# Source Brief: brief_01052.md
# Brief Index: 1052
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Particle:
    def __init__(self, pos, vel, size, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = size
        self.color = color
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.size = max(0, self.size * 0.98)

    def draw(self, surface, offset):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.max_lifespan))
            color_with_alpha = self.color + (alpha,)
            temp_surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (self.size, self.size), self.size)
            render_pos = self.pos - offset
            surface.blit(temp_surf, (int(render_pos.x - self.size), int(render_pos.y - self.size)))

class Projectile:
    def __init__(self, pos, vel, color, damage, owner, size=4):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.color = color
        self.damage = damage
        self.owner = owner  # 'player' or 'enemy'
        self.size = size
        self.lifespan = 120 # 4 seconds at 30fps

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, surface, offset):
        render_pos = self.pos - offset
        start = render_pos - self.vel.normalize() * 10
        pygame.draw.line(surface, self.color, (int(start.x), int(start.y)), (int(render_pos.x), int(render_pos.y)), max(1, int(self.size/2)))
        pygame.gfxdraw.filled_circle(surface, int(render_pos.x), int(render_pos.y), self.size, self.color)
        pygame.gfxdraw.aacircle(surface, int(render_pos.x), int(render_pos.y), self.size, self.color)


class Asteroid:
    def __init__(self, pos, vel, size):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.size = size
        self.health = size * 2
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-2, 2)
        self.color = (200, 200, 100)
        self.points = []
        num_points = max(5, int(size / 3))
        for _ in range(num_points):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(size * 0.7, size)
            self.points.append((math.cos(angle) * radius, math.sin(angle) * radius))

    def update(self):
        self.pos += self.vel
        self.rotation = (self.rotation + self.rotation_speed) % 360

    def draw(self, surface, offset):
        render_pos = self.pos - offset
        rotated_points = []
        rad = math.radians(self.rotation)
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)
        for x, y in self.points:
            rx = x * cos_rad - y * sin_rad + render_pos.x
            ry = x * sin_rad + y * cos_rad + render_pos.y
            rotated_points.append((int(rx), int(ry)))
        
        if len(rotated_points) > 2:
            pygame.gfxdraw.aapolygon(surface, rotated_points, self.color)
            pygame.gfxdraw.filled_polygon(surface, rotated_points, self.color)


class Enemy:
    def __init__(self, pos, health, size, world_bounds):
        self.pos = pygame.math.Vector2(pos)
        self.health = health
        self.max_health = health
        self.size = size
        self.color = (255, 50, 50)
        self.shoot_cooldown = 0
        self.shoot_timer = random.randint(60, 120)
        self.vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * 1.5
        self.world_bounds = world_bounds
        self.pulse = 0

    def update(self, player_pos):
        self.pos += self.vel
        # Simple bounce off world bounds
        if not (self.world_bounds[0] < self.pos.x < self.world_bounds[2]):
            self.vel.x *= -1
        if not (self.world_bounds[1] < self.pos.y < self.world_bounds[3]):
            self.vel.y *= -1
        self.pos.x = max(self.world_bounds[0], min(self.pos.x, self.world_bounds[2]))
        self.pos.y = max(self.world_bounds[1], min(self.pos.y, self.world_bounds[3]))

        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.pulse = (self.pulse + 5) % 360

    def should_shoot(self, player_pos):
        if self.shoot_cooldown == 0 and self.pos.distance_to(player_pos) < 400:
            self.shoot_cooldown = self.shoot_timer
            return True
        return False

    def draw(self, surface, offset):
        render_pos = self.pos - offset
        s = self.size + math.sin(math.radians(self.pulse)) * 3
        points = [
            (render_pos.x, render_pos.y - s),
            (render_pos.x + s, render_pos.y),
            (render_pos.x, render_pos.y + s),
            (render_pos.x - s, render_pos.y)
        ]
        pygame.gfxdraw.aapolygon(surface, points, self.color)
        pygame.gfxdraw.filled_polygon(surface, points, self.color)

class Upgrade:
    def __init__(self, pos, upgrade_type):
        self.pos = pygame.math.Vector2(pos)
        self.type = upgrade_type
        self.size = 12
        self.bob_angle = random.uniform(0, 360)
        self.color_map = {
            'HEALTH_UP': (100, 255, 100),
            'DAMAGE_UP': (255, 150, 50),
            'FIRE_RATE_UP': (100, 100, 255),
            'SECONDARY_WEAPON': (255, 50, 255)
        }
        self.color = self.color_map.get(self.type, (255, 255, 255))
    
    def update(self):
        self.bob_angle = (self.bob_angle + 3) % 360

    def draw(self, surface, offset):
        bob = math.sin(math.radians(self.bob_angle)) * 3
        render_pos = self.pos - offset
        render_pos.y += bob
        
        # Draw a glowing effect
        for i in range(3):
            alpha = 80 - i * 20
            pygame.gfxdraw.aacircle(surface, int(render_pos.x), int(render_pos.y), self.size + i * 4, self.color + (alpha,))

        if self.type == 'HEALTH_UP': # Cross
            pygame.draw.rect(surface, self.color, (render_pos.x - self.size/2, render_pos.y - self.size/6, self.size, self.size/3))
            pygame.draw.rect(surface, self.color, (render_pos.x - self.size/6, render_pos.y - self.size/2, self.size/3, self.size))
        elif self.type == 'DAMAGE_UP': # Up arrow
            pygame.draw.polygon(surface, self.color, [(render_pos.x, render_pos.y - self.size/2), (render_pos.x + self.size/2, render_pos.y), (render_pos.x - self.size/2, render_pos.y)])
        elif self.type == 'FIRE_RATE_UP': # Three dots
            for i in range(-1, 2):
                pygame.gfxdraw.filled_circle(surface, int(render_pos.x + i * self.size/2), int(render_pos.y), 2, self.color)
        elif self.type == 'SECONDARY_WEAPON': # Star
            points = []
            for i in range(5):
                angle = math.radians(i * 72 - 90)
                points.append((render_pos.x + math.cos(angle) * self.size, render_pos.y + math.sin(angle) * self.size))
                angle += math.radians(36)
                points.append((render_pos.x + math.cos(angle) * self.size/2, render_pos.y + math.sin(angle) * self.size/2))
            pygame.gfxdraw.filled_polygon(surface, points, self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Pilot a lone starship through a hostile nebula. Destroy asteroids and enemy fighters "
        "to find upgrades and reach the nebula's core."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to fire your primary weapon and "
        "shift to use your secondary weapon."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_SIZE = 10000

        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.FONT_S = pygame.font.Font(None, 24)
        self.FONT_M = pygame.font.Font(None, 32)
        self.FONT_L = pygame.font.Font(None, 48)

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ASTEROID = (200, 200, 100)
        self.COLOR_PLAYER_PROJ = (100, 200, 255)
        self.COLOR_ENEMY_PROJ = (255, 100, 50)
        self.COLOR_UI = (220, 220, 220)
        
        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = 0
        self.player_max_health = 100
        self.player_speed = 6
        self.player_size = 12
        self.player_upgrades = set()
        self.player_primary_cooldown = 0
        self.player_secondary_cooldown = 0
        self.step_reward = 0

        self.enemies = []
        self.asteroids = []
        self.projectiles = []
        self.particles = []
        self.upgrades = []
        
        self.nebula_core_pos = pygame.math.Vector2(random.uniform(self.WORLD_SIZE*0.8, self.WORLD_SIZE), random.uniform(self.WORLD_SIZE*0.8, self.WORLD_SIZE))
        
        self.starfield = [
            [(random.randint(0, self.WORLD_SIZE), random.randint(0, self.WORLD_SIZE)), random.randint(1, 3)]
            for _ in range(300)
        ]
        
        self.enemy_spawn_timer = 0
        self.asteroid_spawn_timer = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_world_pos = pygame.math.Vector2(self.WORLD_SIZE / 2, self.WORLD_SIZE / 2)
        self.player_health = self.player_max_health
        self.player_upgrades = set()
        
        self.player_primary_cooldown = 0
        self.player_secondary_cooldown = 0
        
        self.enemies.clear()
        self.asteroids.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.upgrades.clear()
        
        self.enemy_spawn_timer = 120
        self.asteroid_spawn_timer = 60

        # Initial spawns
        for _ in range(5):
            self._spawn_asteroid()
        for _ in range(2):
            self._spawn_enemy()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.step_reward = 0
        self.steps += 1
        
        self._handle_input(action)
        self._update_game_state()
        self._handle_collisions()
        self._spawn_entities()
        self._cleanup_entities()
        
        terminated = self._check_termination()
        reward = self.step_reward
        
        if terminated:
            if self.player_health <= 0:
                reward += -100 # Death penalty
            elif self.player_world_pos.distance_to(self.nebula_core_pos) < 200:
                reward += 100 # Victory bonus
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y -= 1 # Up
        elif movement == 2: move_vec.y += 1 # Down
        elif movement == 3: move_vec.x -= 1 # Left
        elif movement == 4: move_vec.x += 1 # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_world_pos += move_vec * self.player_speed

        # Boundary checks
        self.player_world_pos.x = max(0, min(self.player_world_pos.x, self.WORLD_SIZE))
        self.player_world_pos.y = max(0, min(self.player_world_pos.y, self.WORLD_SIZE))

        # Handle shooting
        self.player_primary_cooldown = max(0, self.player_primary_cooldown - 1)
        self.player_secondary_cooldown = max(0, self.player_secondary_cooldown - 1)
        
        fire_rate_mod = 0.75 if 'FIRE_RATE_UP' in self.player_upgrades else 1.0
        
        if space_held and self.player_primary_cooldown == 0:
            self._fire_primary()
            self.player_primary_cooldown = 15 * fire_rate_mod
        
        if shift_held and 'SECONDARY_WEAPON' in self.player_upgrades and self.player_secondary_cooldown == 0:
            self._fire_secondary()
            self.player_secondary_cooldown = 45 * fire_rate_mod
    
    def _fire_primary(self):
        # sfx: Player laser shot
        damage_mod = 1.5 if 'DAMAGE_UP' in self.player_upgrades else 1.0
        mouse_pos = pygame.mouse.get_pos()
        direction = (pygame.math.Vector2(mouse_pos) - self.player_pos).normalize()
        
        vel = direction * 12
        proj = Projectile(self.player_world_pos, vel, self.COLOR_PLAYER_PROJ, 10 * damage_mod, 'player')
        self.projectiles.append(proj)

    def _fire_secondary(self):
        # sfx: Player spread shot
        damage_mod = 1.5 if 'DAMAGE_UP' in self.player_upgrades else 1.0
        mouse_pos = pygame.mouse.get_pos()
        base_direction = (pygame.math.Vector2(mouse_pos) - self.player_pos).normalize()

        for i in range(-1, 2):
            angle = i * 15 # degrees
            direction = base_direction.rotate(angle)
            vel = direction * 10
            proj = Projectile(self.player_world_pos, vel, (255, 50, 255), 7 * damage_mod, 'player', size=5)
            self.projectiles.append(proj)

    def _update_game_state(self):
        for p in self.particles: p.update()
        for u in self.upgrades: u.update()
        for a in self.asteroids: a.update()
        for e in self.enemies:
            e.update(self.player_world_pos)
            if e.should_shoot(self.player_world_pos):
                # sfx: Enemy laser shot
                direction = (self.player_world_pos - e.pos).normalize()
                vel = direction * 8
                self.projectiles.append(Projectile(e.pos, vel, self.COLOR_ENEMY_PROJ, 10, 'enemy'))
        for p in self.projectiles: p.update()

    def _handle_collisions(self):
        # Player projectiles vs entities
        for proj in self.projectiles[:]:
            if proj.owner != 'player': continue
            
            # Vs Asteroids
            for ast in self.asteroids[:]:
                if proj.pos.distance_to(ast.pos) < ast.size + proj.size:
                    self._create_explosion(proj.pos, self.COLOR_ASTEROID, 5)
                    ast.health -= proj.damage
                    if ast.health <= 0:
                        self.step_reward += 0.1
                        self.score += 10
                        self._create_explosion(ast.pos, self.COLOR_ASTEROID, int(ast.size / 2))
                        self.asteroids.remove(ast)
                        # sfx: Asteroid explosion
                    if proj in self.projectiles: self.projectiles.remove(proj)
                    break
            else: # Vs Enemies
                for enemy in self.enemies[:]:
                    if proj.pos.distance_to(enemy.pos) < enemy.size + proj.size:
                        self._create_explosion(proj.pos, self.COLOR_ENEMY, 5)
                        enemy.health -= proj.damage
                        if enemy.health <= 0:
                            self.step_reward += 1.0
                            self.score += 100
                            self._create_explosion(enemy.pos, self.COLOR_ENEMY, 15)
                            # Chance to drop an upgrade
                            if random.random() < 0.2:
                                self._spawn_upgrade(enemy.pos)
                            self.enemies.remove(enemy)
                            # sfx: Enemy explosion
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        break

        # Enemy projectiles vs Player
        for proj in self.projectiles[:]:
            if proj.owner != 'enemy': continue
            if proj.pos.distance_to(self.player_world_pos) < self.player_size + proj.size:
                self.player_health -= proj.damage
                self.step_reward -= 0.5
                self._create_explosion(self.player_world_pos, self.COLOR_PLAYER, 10, is_player_offset=True)
                # sfx: Player hit
                if proj in self.projectiles: self.projectiles.remove(proj)

        # Player vs Entities
        # Vs Asteroids
        for ast in self.asteroids[:]:
            if self.player_world_pos.distance_to(ast.pos) < self.player_size + ast.size:
                self.player_health -= 15
                self.step_reward -= 0.5
                self._create_explosion(self.player_world_pos, self.COLOR_PLAYER, 10, is_player_offset=True)
                self._create_explosion(ast.pos, self.COLOR_ASTEROID, int(ast.size / 2))
                self.asteroids.remove(ast)

        # Vs Enemies
        for enemy in self.enemies[:]:
            if self.player_world_pos.distance_to(enemy.pos) < self.player_size + enemy.size:
                self.player_health -= 25
                self.step_reward -= 0.5
                self._create_explosion(self.player_world_pos, self.COLOR_PLAYER, 10, is_player_offset=True)
                self._create_explosion(enemy.pos, self.COLOR_ENEMY, 15)
                self.enemies.remove(enemy)

        # Vs Upgrades
        for upg in self.upgrades[:]:
            if self.player_world_pos.distance_to(upg.pos) < self.player_size + upg.size:
                self.step_reward += 5.0
                self.score += 250
                self._apply_upgrade(upg.type)
                # sfx: Upgrade pickup
                self.upgrades.remove(upg)

    def _apply_upgrade(self, upgrade_type):
        if upgrade_type == 'HEALTH_UP':
            self.player_health = min(self.player_max_health, self.player_health + 40)
        elif upgrade_type not in self.player_upgrades:
            self.player_upgrades.add(upgrade_type)

    def _spawn_entities(self):
        # Difficulty scaling
        enemy_spawn_mod = 0.95 ** (self.steps // 500)
        asteroid_spawn_mod = 0.98 ** (self.steps // 250)

        self.enemy_spawn_timer = max(1, self.enemy_spawn_timer - 1)
        self.asteroid_spawn_timer = max(1, self.asteroid_spawn_timer - 1)

        if self.enemy_spawn_timer == 0:
            self._spawn_enemy()
            self.enemy_spawn_timer = int(180 * enemy_spawn_mod)

        if self.asteroid_spawn_timer == 0:
            self._spawn_asteroid()
            self.asteroid_spawn_timer = int(75 * asteroid_spawn_mod)

    def _spawn_enemy(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        spawn_dist = 500
        if edge == 'top': pos = (random.uniform(self.player_world_pos.x - spawn_dist, self.player_world_pos.x + spawn_dist), self.player_world_pos.y - spawn_dist)
        elif edge == 'bottom': pos = (random.uniform(self.player_world_pos.x - spawn_dist, self.player_world_pos.x + spawn_dist), self.player_world_pos.y + spawn_dist)
        elif edge == 'left': pos = (self.player_world_pos.x - spawn_dist, random.uniform(self.player_world_pos.y - spawn_dist, self.player_world_pos.y + spawn_dist))
        else: pos = (self.player_world_pos.x + spawn_dist, random.uniform(self.player_world_pos.y - spawn_dist, self.player_world_pos.y + spawn_dist))
        
        health_mod = 1.05 ** (self.steps // 500)
        health = 20 * health_mod
        bounds = (self.player_world_pos.x - 800, self.player_world_pos.y - 600, self.player_world_pos.x + 800, self.player_world_pos.y + 600)
        self.enemies.append(Enemy(pos, health, 15, bounds))

    def _spawn_asteroid(self):
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        spawn_dist = 600
        if edge == 'top': pos = (random.uniform(self.player_world_pos.x - spawn_dist, self.player_world_pos.x + spawn_dist), self.player_world_pos.y - spawn_dist)
        elif edge == 'bottom': pos = (random.uniform(self.player_world_pos.x - spawn_dist, self.player_world_pos.x + spawn_dist), self.player_world_pos.y + spawn_dist)
        elif edge == 'left': pos = (self.player_world_pos.x - spawn_dist, random.uniform(self.player_world_pos.y - spawn_dist, self.player_world_pos.y + spawn_dist))
        else: pos = (self.player_world_pos.x + spawn_dist, random.uniform(self.player_world_pos.y - spawn_dist, self.player_world_pos.y + spawn_dist))

        vel = (self.player_world_pos - pygame.math.Vector2(pos)).normalize() * random.uniform(0.5, 1.5)
        size = random.randint(15, 40)
        self.asteroids.append(Asteroid(pos, vel, size))

    def _spawn_upgrade(self, pos):
        upgrade_type = random.choice(['HEALTH_UP', 'DAMAGE_UP', 'FIRE_RATE_UP', 'SECONDARY_WEAPON'])
        if upgrade_type in self.player_upgrades: # Avoid duplicate non-stacking upgrades
            upgrade_type = 'HEALTH_UP'
        self.upgrades.append(Upgrade(pos, upgrade_type))

    def _cleanup_entities(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        self.projectiles = [p for p in self.projectiles if p.lifespan > 0 and 0 < p.pos.x < self.WORLD_SIZE and 0 < p.pos.y < self.WORLD_SIZE]
        
        # Remove entities far from player
        cull_dist_sq = 1000**2
        self.asteroids = [a for a in self.asteroids if a.pos.distance_squared_to(self.player_world_pos) < cull_dist_sq]
        self.enemies = [e for e in self.enemies if e.pos.distance_squared_to(self.player_world_pos) < cull_dist_sq]
        self.upgrades = [u for u in self.upgrades if u.pos.distance_squared_to(self.player_world_pos) < cull_dist_sq]

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        if self.steps >= 5000:
            self.game_over = True
            return True
        if self.player_world_pos.distance_to(self.nebula_core_pos) < 200:
            self.game_over = True
            return True
        return False

    def _create_explosion(self, pos, color, num_particles, is_player_offset=False):
        final_pos = pos
        if is_player_offset:
            offset = (self.player_world_pos - pos).normalize() * self.player_size
            final_pos = pos + offset
        
        for _ in range(num_particles):
            vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(1, 4)
            size = random.uniform(2, 6)
            lifespan = random.randint(20, 40)
            self.particles.append(Particle(final_pos, vel, size, color, lifespan))
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        camera_offset = self.player_world_pos - self.player_pos

        # Render starfield (parallax)
        for i, (pos, size) in enumerate(self.starfield):
            parallax_factor = size / 3.0
            render_x = (pos[0] - camera_offset.x * parallax_factor) % self.WIDTH
            render_y = (pos[1] - camera_offset.y * parallax_factor) % self.HEIGHT
            color_val = 50 + int(size * 30)
            color = (color_val, color_val, color_val + 20)
            pygame.draw.circle(self.screen, color, (int(render_x), int(render_y)), size)

        # Render core marker
        core_dir = self.nebula_core_pos - self.player_world_pos
        if core_dir.length() > 0:
            core_dir.normalize_ip()
            indicator_pos = self.player_pos + core_dir * 50
            pygame.draw.line(self.screen, (255, 255, 0, 100), self.player_pos, indicator_pos, 2)
            pygame.gfxdraw.filled_circle(self.screen, int(indicator_pos.x), int(indicator_pos.y), 4, (255, 255, 0))

        # Render entities
        for p in self.particles: p.draw(self.screen, camera_offset)
        for u in self.upgrades: u.draw(self.screen, camera_offset)
        for a in self.asteroids: a.draw(self.screen, camera_offset)
        for e in self.enemies: e.draw(self.screen, camera_offset)
        
        # Render Player
        p_size = self.player_size
        p1 = self.player_pos + pygame.math.Vector2(0, -p_size * 1.2)
        p2 = self.player_pos + pygame.math.Vector2(-p_size, p_size * 0.8)
        p3 = self.player_pos + pygame.math.Vector2(p_size, p_size * 0.8)
        
        # Glow effect
        for i in range(4):
            alpha = 100 - i * 25
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), p_size + i * 4, self.COLOR_PLAYER + (alpha,))
        
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

        for p in self.projectiles: p.draw(self.screen, camera_offset)

    def _render_ui(self):
        # Score
        score_text = self.FONT_M.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Health Bar
        health_ratio = max(0, self.player_health / self.player_max_health)
        bar_width = 200
        bar_height = 20
        health_bar_rect = pygame.Rect(10, 10, bar_width, bar_height)
        health_fill_rect = pygame.Rect(10, 10, int(bar_width * health_ratio), bar_height)
        
        pygame.draw.rect(self.screen, (100, 0, 0), health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, health_fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI, health_bar_rect, 2)

        # Upgrade Icons
        icon_size = 24
        icon_spacing = 30
        start_x = self.WIDTH/2 - (len(self.player_upgrades) * icon_spacing)/2
        for i, upg_type in enumerate(self.player_upgrades):
            icon_rect = pygame.Rect(start_x + i * icon_spacing, self.HEIGHT - 40, icon_size, icon_size)
            upg_obj = Upgrade(pygame.math.Vector2(0,0), upg_type) # Dummy for color
            pygame.draw.rect(self.screen, upg_obj.color, icon_rect, 0, 4)
            pygame.draw.rect(self.screen, self.COLOR_UI, icon_rect, 2, 4)
            
            # Simple letter indicator
            letter = self.FONT_S.render(upg_type[0], True, (0,0,0))
            self.screen.blit(letter, (icon_rect.centerx - letter.get_width()/2, icon_rect.centery - letter.get_height()/2))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_health <= 0:
                end_text = self.FONT_L.render("SHIP DESTROYED", True, self.COLOR_ENEMY)
            else:
                end_text = self.FONT_L.render("NEBULA CORE REACHED", True, self.COLOR_PLAYER)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))

    def close(self):
        pygame.quit()

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually:
    pygame.display.set_caption("Nebula Core")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Convert obs for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Lock to 30 FPS for manual play
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    
    env.close()