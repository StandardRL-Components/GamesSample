import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T19:20:36.108954
# Source Brief: brief_03157.md
# Brief Index: 3157
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# --- Constants ---
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 400
FPS = 30
MAX_EPISODE_STEPS = 7500

# --- Colors ---
COLOR_BG = (10, 5, 25)
COLOR_PLAYER = (0, 255, 128)
COLOR_PLAYER_GLOW = (0, 255, 128, 50)
COLOR_ENEMY_SWARMER = (255, 50, 50)
COLOR_ENEMY_BRUISER = (255, 100, 50)
COLOR_ENEMY_SHIELDER = (200, 50, 255)
COLOR_ENEMY_BOSS = (255, 255, 0)
COLOR_ENEMY_GLOW = (255, 50, 50, 50)
COLOR_PLAYER_BULLET = (0, 200, 255)
COLOR_PLAYER_BULLET_GLOW = (0, 200, 255, 100)
COLOR_ENEMY_BULLET = (255, 150, 0)
COLOR_ENEMY_BULLET_GLOW = (255, 150, 0, 100)
COLOR_SALVAGE = (255, 220, 0)
COLOR_SALVAGE_GLOW = (255, 220, 0, 80)
COLOR_SHIELD = (100, 150, 255, 128)
COLOR_WHITE = (240, 240, 240)
COLOR_UI_BG = (30, 30, 60, 180)
COLOR_HEALTH = (0, 255, 0)
COLOR_ENERGY = (0, 180, 255)
COLOR_UI_TEXT = (220, 220, 255)

# --- Game Object Classes ---

class Player:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = SCREEN_WIDTH / 2
        self.y = SCREEN_HEIGHT / 2
        self.vx = 0
        self.vy = 0
        self.size = 12
        self.speed = 1.5
        self.damping = 0.85
        self.max_health = 100
        self.health = self.max_health
        self.max_energy = 100
        self.energy = self.max_energy
        self.energy_regen = 0.5
        self.weapon_cooldown = 0
        self.active_weapon_idx = 0
        self.unlocked_weapons = ["PULSE_LASER"]

    def update(self, movement):
        if movement == 1: self.vy -= self.speed
        if movement == 2: self.vy += self.speed
        if movement == 3: self.vx -= self.speed
        if movement == 4: self.vx += self.speed

        self.vx *= self.damping
        self.vy *= self.damping
        self.x += self.vx
        self.y += self.vy

        self.x = np.clip(self.x, self.size, SCREEN_WIDTH - self.size)
        self.y = np.clip(self.y, self.size, SCREEN_HEIGHT - self.size)

        self.energy = min(self.max_energy, self.energy + self.energy_regen)
        if self.weapon_cooldown > 0:
            self.weapon_cooldown -= 1

    def take_damage(self, amount):
        self.health -= amount
        self.health = max(0, self.health)
        return amount

    def draw(self, surface):
        points = [
            (self.x, self.y - self.size),
            (self.x - self.size / 1.5, self.y + self.size / 2),
            (self.x + self.size / 1.5, self.y + self.size / 2),
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]

        # Glow effect
        pygame.gfxdraw.filled_trigon(surface, *int_points[0], *int_points[1], *int_points[2], COLOR_PLAYER_GLOW)
        # Ship body
        pygame.gfxdraw.filled_trigon(surface, *int_points[0], *int_points[1], *int_points[2], COLOR_PLAYER)
        pygame.gfxdraw.aatrigon(surface, *int_points[0], *int_points[1], *int_points[2], COLOR_PLAYER)


class Enemy:
    def __init__(self, x, y, player, health_mult, damage_mult):
        self.x = x
        self.y = y
        self.player = player
        self.health_mult = health_mult
        self.damage_mult = damage_mult
        self.cooldown = random.randint(30, 90)
        self.is_boss = False

    def update(self, enemy_projectiles):
        self._move()
        self.x = np.clip(self.x, self.size, SCREEN_WIDTH - self.size)
        self.y = np.clip(self.y, self.size, SCREEN_HEIGHT - self.size)
        
        self.cooldown -= 1
        if self.cooldown <= 0:
            self._fire(enemy_projectiles)

    def _move(self):
        pass

    def _fire(self, enemy_projectiles):
        pass

    def take_damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def draw(self, surface):
        # Health bar
        if self.health < self.max_health:
            bar_width = 30
            bar_height = 4
            health_pct = self.health / self.max_health
            pygame.draw.rect(surface, (100, 0, 0), (int(self.x - bar_width/2), int(self.y - self.size - 10), bar_width, bar_height))
            pygame.draw.rect(surface, (255, 0, 0), (int(self.x - bar_width/2), int(self.y - self.size - 10), int(bar_width * health_pct), bar_height))

class Swarmer(Enemy):
    def __init__(self, x, y, player, health_mult, damage_mult):
        super().__init__(x, y, player, health_mult, damage_mult)
        self.size = 8
        self.max_health = 10 * health_mult
        self.health = self.max_health
        self.speed = random.uniform(1.0, 1.5)
        self.angle = random.uniform(0, 2 * math.pi)
        self.turn_speed = random.uniform(0.02, 0.05)
        self.fire_rate = 60

    def _move(self):
        target_angle = math.atan2(self.player.y - self.y, self.player.x - self.x)
        self.angle += self.turn_speed * ((target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def _fire(self, enemy_projectiles):
        angle_to_player = math.atan2(self.player.y - self.y, self.player.x - self.x)
        enemy_projectiles.append(Projectile(self.x, self.y, angle_to_player, 4, 5 * self.damage_mult, False))
        self.cooldown = self.fire_rate
        # sfx: enemy_laser_light.wav

    def draw(self, surface):
        points = [
            (self.x + math.cos(self.angle) * self.size, self.y + math.sin(self.angle) * self.size),
            (self.x + math.cos(self.angle + 2.3) * self.size, self.y + math.sin(self.angle + 2.3) * self.size),
            (self.x + math.cos(self.angle - 2.3) * self.size, self.y + math.sin(self.angle - 2.3) * self.size),
        ]
        int_points = [(int(p[0]), int(p[1])) for p in points]
        pygame.gfxdraw.filled_trigon(surface, *int_points[0], *int_points[1], *int_points[2], COLOR_ENEMY_GLOW)
        pygame.gfxdraw.filled_trigon(surface, *int_points[0], *int_points[1], *int_points[2], COLOR_ENEMY_SWARMER)
        super().draw(surface)

class Bruiser(Enemy):
    def __init__(self, x, y, player, health_mult, damage_mult):
        super().__init__(x, y, player, health_mult, damage_mult)
        self.size = 15
        self.max_health = 50 * health_mult
        self.health = self.max_health
        self.speed = 0.5
        self.fire_rate = 120

    def _move(self):
        dist_to_player = math.hypot(self.player.x - self.x, self.player.y - self.y)
        if dist_to_player > 200:
            angle_to_player = math.atan2(self.player.y - self.y, self.player.x - self.x)
            self.x += math.cos(angle_to_player) * self.speed
            self.y += math.sin(angle_to_player) * self.speed

    def _fire(self, enemy_projectiles):
        angle_to_player = math.atan2(self.player.y - self.y, self.player.x - self.x)
        enemy_projectiles.append(Projectile(self.x, self.y, angle_to_player, 3, 20 * self.damage_mult, False, size=5))
        self.cooldown = self.fire_rate
        # sfx: enemy_laser_heavy.wav

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, COLOR_ENEMY_GLOW)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, COLOR_ENEMY_BRUISER)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size, COLOR_ENEMY_BRUISER)
        super().draw(surface)

class Shielder(Enemy):
    def __init__(self, x, y, player, health_mult, damage_mult):
        super().__init__(x, y, player, health_mult, damage_mult)
        self.size = 12
        self.max_health = 30 * health_mult
        self.health = self.max_health
        self.fire_rate = 90
        self.move_timer = 0
        self.move_target = (self.x, self.y)
        self.is_shielded = False
        self.shield_timer = 0
        self.shield_cooldown = 300
        self.shield_duration = 120

    def _move(self):
        self.move_timer -= 1
        if self.move_timer <= 0:
            self.move_target = (random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50))
            self.move_timer = random.randint(60, 120)

        angle = math.atan2(self.move_target[1] - self.y, self.move_target[0] - self.x)
        self.x += math.cos(angle) * 1.0
        self.y += math.sin(angle) * 1.0

    def update(self, enemy_projectiles):
        super().update(enemy_projectiles)
        if self.shield_cooldown > 0:
            self.shield_cooldown -= 1
        else:
            self.is_shielded = True
            self.shield_timer = self.shield_duration
            self.shield_cooldown = 300 + self.shield_duration

        if self.is_shielded:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.is_shielded = False

    def _fire(self, enemy_projectiles):
        for i in range(3):
            angle = math.atan2(self.player.y - self.y, self.player.x - self.x) + (i - 1) * 0.3
            enemy_projectiles.append(Projectile(self.x, self.y, angle, 3.5, 8 * self.damage_mult, False))
        self.cooldown = self.fire_rate
        # sfx: enemy_laser_multi.wav

    def take_damage(self, amount):
        if self.is_shielded:
            return False
        return super().take_damage(amount)

    def draw(self, surface):
        s = pygame.Surface((self.size * 3, self.size * 3), pygame.SRCALPHA)
        if self.is_shielded:
            pygame.draw.circle(s, COLOR_SHIELD, (self.size*1.5, self.size*1.5), self.size*1.5)
        
        rect = pygame.Rect(self.size*0.5, self.size*0.5, self.size*2, self.size*2)
        pygame.draw.rect(s, COLOR_ENEMY_SHIELDER, rect)
        surface.blit(s, (int(self.x - self.size*1.5), int(self.y - self.size*1.5)))
        super().draw(surface)

class Boss(Enemy):
    def __init__(self, x, y, player, health_mult, damage_mult, boss_level):
        super().__init__(x, y, player, health_mult, damage_mult)
        self.size = 30 + boss_level * 10
        self.max_health = (250 + boss_level * 250) * health_mult
        self.health = self.max_health
        self.is_boss = True
        self.boss_level = boss_level # 0, 1, 2
        self.phase = 0
        self.phase_timer = 0
        self.angle = 0

    def _move(self):
        if self.boss_level == 0: # Sentry boss
            self.x, self.y = SCREEN_WIDTH/2, 100
        elif self.boss_level == 1: # Moving boss
            self.x = SCREEN_WIDTH/2 + math.sin(self.phase_timer * 0.01) * 200
            self.y = 100 + math.cos(self.phase_timer * 0.02) * 50
        else: # Final boss
            self.x = SCREEN_WIDTH/2 + math.sin(self.phase_timer * 0.015) * 250
            self.y = SCREEN_HEIGHT/2 + math.cos(self.phase_timer * 0.01) * 150

    def update(self, enemy_projectiles):
        self._move()
        self.phase_timer += 1
        self.cooldown -= 1
        if self.cooldown <= 0:
            self._fire(enemy_projectiles)
            
        if self.health < self.max_health / 2 and self.phase == 0:
            self.phase = 1
            # sfx: boss_phase_change.wav

    def _fire(self, enemy_projectiles):
        if self.boss_level == 0:
            self.angle += 0.1
            for i in range(5):
                angle = self.angle + i * (2 * math.pi / 5)
                enemy_projectiles.append(Projectile(self.x, self.y, angle, 3, 10 * self.damage_mult, False))
            self.cooldown = 45
        elif self.boss_level == 1:
            angle_to_player = math.atan2(self.player.y - self.y, self.player.x - self.x)
            for i in range(-2, 3):
                enemy_projectiles.append(Projectile(self.x, self.y, angle_to_player + i * 0.2, 4, 15 * self.damage_mult, False, size=4))
            self.cooldown = 90
            if self.phase == 1:
                self.cooldown = 60
        else: # Final boss
            self.angle += 0.03
            if self.phase_timer % 120 < 60:
                for i in range(8):
                    angle = self.angle + i * (2 * math.pi / 8)
                    enemy_projectiles.append(Projectile(self.x, self.y, angle, 3 + self.phase, 12 * self.damage_mult, False))
                self.cooldown = 15
            else:
                angle_to_player = math.atan2(self.player.y - self.y, self.player.x - self.x)
                for i in range(-3, 4):
                    enemy_projectiles.append(Projectile(self.x, self.y, angle_to_player + i * 0.15, 5, 20 * self.damage_mult, False, size=5))
                self.cooldown = 120
        # sfx: boss_fire.wav
        
    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, COLOR_ENEMY_GLOW)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, COLOR_ENEMY_BOSS)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size, COLOR_ENEMY_BOSS)

        # Inner rotating element
        for i in range(5):
            angle = self.phase_timer * 0.05 + i * (2 * math.pi / 5)
            px = self.x + math.cos(angle) * self.size * 0.6
            py = self.y + math.sin(angle) * self.size * 0.6
            pygame.draw.circle(surface, COLOR_ENEMY_SWARMER, (int(px), int(py)), 5)
        super().draw(surface)


class Projectile:
    def __init__(self, x, y, angle, speed, damage, is_player_bullet, size=3):
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.damage = damage
        self.is_player_bullet = is_player_bullet
        self.size = size
        self.lifetime = (SCREEN_WIDTH + SCREEN_HEIGHT) / speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        return self.x < 0 or self.x > SCREEN_WIDTH or self.y < 0 or self.y > SCREEN_HEIGHT or self.lifetime <= 0

    def draw(self, surface):
        color = COLOR_PLAYER_BULLET if self.is_player_bullet else COLOR_ENEMY_BULLET
        glow_color = COLOR_PLAYER_BULLET_GLOW if self.is_player_bullet else COLOR_ENEMY_BULLET_GLOW
        
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size + 2, glow_color)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size, color)


class Particle:
    def __init__(self, x, y, color, lifetime, size, speed):
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        angle = random.uniform(0, 2 * math.pi)
        vel = random.uniform(0, speed)
        self.vx = math.cos(angle) * vel
        self.vy = math.sin(angle) * vel

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.lifetime -= 1
        return self.lifetime <= 0

    def draw(self, surface):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            current_size = int(self.size * (self.lifetime / self.max_lifetime))
            if current_size > 0:
                color = (*self.color[:3], alpha)
                s = pygame.Surface((current_size*2, current_size*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (current_size, current_size), current_size)
                surface.blit(s, (int(self.x - current_size), int(self.y - current_size)))

class Salvage:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 6
        self.value = 1

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size+3, COLOR_SALVAGE_GLOW)
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.size, COLOR_SALVAGE)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.size, COLOR_SALVAGE)

# --- Main Environment Class ---

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Survive waves of alien ships in this top-down arena shooter. "
        "Collect salvage from fallen enemies to upgrade your arsenal and take on powerful bosses."
    )
    user_guide = (
        "Controls: Use arrow keys or WASD to move. Press space to fire your weapon and shift to switch between unlocked weapons."
    )
    auto_advance = True
    
    WEAPON_SYSTEMS = {
        "PULSE_LASER": {"cost": 5, "cooldown": 8, "damage": 8, "speed": 8},
        "PLASMA_CANNON": {"cost": 30, "cooldown": 45, "damage": 40, "speed": 5, "size": 6},
        "SPREAD_SHOT": {"cost": 20, "cooldown": 30, "damage": 5, "speed": 7, "count": 3, "spread": 0.4},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("sans-serif", 16)
        self.font_large = pygame.font.SysFont("sans-serif", 24)
        
        self.render_mode = render_mode
        self._initialize_state_variables()
        self.reset()
        
    
    def _initialize_state_variables(self):
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = Player()
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.salvage_items = []
        self.stars = []
        self.shift_pressed_last_frame = False
        self.enemy_spawn_timer = 0
        self.difficulty_tier = 0
        self.health_mult = 1.0
        self.damage_mult = 1.0
        self.boss_spawn_flags = [False, False, False]
        self.boss_active = False
        self.final_boss_defeated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()
        self.player.reset()

        # Create starfield
        for _ in range(150):
            self.stars.append({
                "x": random.uniform(0, SCREEN_WIDTH),
                "y": random.uniform(0, SCREEN_HEIGHT),
                "speed": random.uniform(0.1, 0.5),
                "size": random.uniform(0.5, 1.5),
            })
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward_this_step = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1

        # --- Update Game Logic ---
        self._update_difficulty()
        self._spawn_enemies()

        self.player.update(movement)
        self._handle_player_actions(space_held, shift_held)

        for p in self.particles[:]:
            if p.update(): self.particles.remove(p)
        for proj in self.player_projectiles[:]:
            if proj.update(): self.player_projectiles.remove(proj)
        for proj in self.enemy_projectiles[:]:
            if proj.update(): self.enemy_projectiles.remove(proj)
        
        for enemy in self.enemies[:]:
            enemy.update(self.enemy_projectiles)
        
        # --- Handle Collisions & Rewards ---
        reward_this_step += self._handle_collisions()

        # --- Check Termination ---
        terminated = self.player.health <= 0 or self.steps >= MAX_EPISODE_STEPS or self.final_boss_defeated
        truncated = self.steps >= MAX_EPISODE_STEPS

        if terminated or truncated:
            self.game_over = True
            if self.player.health <= 0:
                reward_this_step -= 100
                self._create_explosion(self.player.x, self.player.y, COLOR_PLAYER, 100)
            if self.final_boss_defeated:
                reward_this_step += 100

        self.score += reward_this_step
        
        return self._get_observation(), reward_this_step, terminated, truncated, self._get_info()

    def _update_difficulty(self):
        new_tier = self.steps // 500
        if new_tier > self.difficulty_tier:
            self.difficulty_tier = new_tier
            self.health_mult = 1.0 + 0.05 * new_tier
            self.damage_mult = 1.0 + 0.05 * new_tier
    
    def _spawn_enemies(self):
        if self.boss_active:
            return
            
        # Check for boss spawns
        boss_steps = [500, 2500, 5000]
        for i, step_req in enumerate(boss_steps):
            if self.steps >= step_req and not self.boss_spawn_flags[i]:
                self.enemies.clear() # Clear regular enemies for boss fight
                self.boss_spawn_flags[i] = True
                self.boss_active = True
                boss = Boss(SCREEN_WIDTH/2, 100, self.player, self.health_mult, self.damage_mult, i)
                self.enemies.append(boss)
                # sfx: boss_spawn.wav
                return

        self.enemy_spawn_timer -= 1
        if self.enemy_spawn_timer <= 0:
            max_enemies = 3 + self.steps // 1000
            if len(self.enemies) < max_enemies:
                spawn_edge = random.choice(['top', 'bottom', 'left', 'right'])
                if spawn_edge == 'top': x, y = random.uniform(0, SCREEN_WIDTH), -20
                elif spawn_edge == 'bottom': x, y = random.uniform(0, SCREEN_WIDTH), SCREEN_HEIGHT + 20
                elif spawn_edge == 'left': x, y = -20, random.uniform(0, SCREEN_HEIGHT)
                else: x, y = SCREEN_WIDTH + 20, random.uniform(0, SCREEN_HEIGHT)

                enemy_type = random.choice([Swarmer])
                if self.steps > 1000: enemy_type = random.choice([Swarmer, Bruiser])
                if self.steps > 2000: enemy_type = random.choice([Swarmer, Bruiser, Shielder])
                
                self.enemies.append(enemy_type(x, y, self.player, self.health_mult, self.damage_mult))
            self.enemy_spawn_timer = max(15, 60 - self.steps // 200)

    def _handle_player_actions(self, space_held, shift_held):
        # Weapon Firing
        weapon_name = self.player.unlocked_weapons[self.player.active_weapon_idx]
        weapon = self.WEAPON_SYSTEMS[weapon_name]
        if space_held and self.player.weapon_cooldown <= 0 and self.player.energy >= weapon["cost"]:
            self.player.energy -= weapon["cost"]
            self.player.weapon_cooldown = weapon["cooldown"]
            # sfx: player_shoot.wav
            
            angle_to_mouse = math.atan2(self.player.y - self.player.y-1, self.player.x - self.player.x) # Shoots straight up
            
            if weapon_name == "PULSE_LASER":
                self.player_projectiles.append(Projectile(self.player.x, self.player.y, -math.pi/2, weapon["speed"], weapon["damage"], True))
            elif weapon_name == "PLASMA_CANNON":
                self.player_projectiles.append(Projectile(self.player.x, self.player.y, -math.pi/2, weapon["speed"], weapon["damage"], True, size=weapon["size"]))
            elif weapon_name == "SPREAD_SHOT":
                for i in range(weapon["count"]):
                    spread_angle = -math.pi/2 + (i - (weapon["count"]-1)/2) * weapon["spread"]
                    self.player_projectiles.append(Projectile(self.player.x, self.player.y, spread_angle, weapon["speed"], weapon["damage"], True))

        # Weapon Switching
        if shift_held and not self.shift_pressed_last_frame:
            self.player.active_weapon_idx = (self.player.active_weapon_idx + 1) % len(self.player.unlocked_weapons)
            # sfx: weapon_switch.wav
        self.shift_pressed_last_frame = shift_held
    
    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                dist = math.hypot(proj.x - enemy.x, proj.y - enemy.y)
                if dist < enemy.size + proj.size:
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    is_destroyed = enemy.take_damage(proj.damage)
                    self._create_explosion(proj.x, proj.y, COLOR_PLAYER_BULLET, 5, size=2, speed=2)
                    # sfx: hit_marker.wav
                    if is_destroyed:
                        reward += 0.1
                        self._create_explosion(enemy.x, enemy.y, COLOR_ENEMY_SWARMER, 50)
                        self.salvage_items.append(Salvage(enemy.x, enemy.y))
                        if enemy.is_boss:
                            reward += 5.0
                            self.boss_active = False
                            if enemy.boss_level == 0 and "PLASMA_CANNON" not in self.player.unlocked_weapons:
                                self.player.unlocked_weapons.append("PLASMA_CANNON")
                            if enemy.boss_level == 1 and "SPREAD_SHOT" not in self.player.unlocked_weapons:
                                self.player.unlocked_weapons.append("SPREAD_SHOT")
                            if enemy.boss_level == 2:
                                self.final_boss_defeated = True
                        self.enemies.remove(enemy)
                        # sfx: explosion.wav
                    break
        
        # Enemy projectiles vs Player
        for proj in self.enemy_projectiles[:]:
            dist = math.hypot(proj.x - self.player.x, proj.y - self.player.y)
            if dist < self.player.size + proj.size:
                self.enemy_projectiles.remove(proj)
                damage_taken = self.player.take_damage(proj.damage)
                reward -= 0.1
                self._create_explosion(proj.x, proj.y, COLOR_ENEMY_BULLET, 15, size=3, speed=3)
                # sfx: player_hit.wav
        
        # Player vs Salvage
        for item in self.salvage_items[:]:
            dist = math.hypot(item.x - self.player.x, item.y - self.player.y)
            if dist < self.player.size + item.size:
                reward += item.value
                self.salvage_items.remove(item)
                # sfx: collect_salvage.wav
        
        return reward
    
    def _create_explosion(self, x, y, color, num_particles, size=4, speed=4):
        for _ in range(num_particles):
            self.particles.append(Particle(x, y, color, random.randint(15, 30), random.uniform(1, size), speed))

    def _get_observation(self):
        self.screen.fill(COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Stars
        for star in self.stars:
            star['y'] += star['speed']
            if star['y'] > SCREEN_HEIGHT:
                star['y'] = 0
                star['x'] = random.uniform(0, SCREEN_WIDTH)
            pygame.draw.circle(self.screen, (200, 200, 255), (int(star['x']), int(star['y'])), star['size'])

        # Game objects
        for item in self.salvage_items: item.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        self.player.draw(self.screen)
        for proj in self.player_projectiles: proj.draw(self.screen)
        for proj in self.enemy_projectiles: proj.draw(self.screen)
        for p in self.particles: p.draw(self.screen)

    def _render_ui(self):
        # Player stats panel
        ui_panel = pygame.Surface((180, 80), pygame.SRCALPHA)
        ui_panel.fill(COLOR_UI_BG)
        
        # Health bar
        health_pct = self.player.health / self.player.max_health
        pygame.draw.rect(ui_panel, (50,0,0), (10, 10, 160, 15))
        pygame.draw.rect(ui_panel, COLOR_HEALTH, (10, 10, int(160 * health_pct), 15))
        health_text = self.font_small.render(f"HP: {int(self.player.health)}/{self.player.max_health}", True, COLOR_UI_TEXT)
        ui_panel.blit(health_text, (15, 11))

        # Energy bar
        energy_pct = self.player.energy / self.player.max_energy
        pygame.draw.rect(ui_panel, (0,0,50), (10, 30, 160, 15))
        pygame.draw.rect(ui_panel, COLOR_ENERGY, (10, 30, int(160 * energy_pct), 15))
        energy_text = self.font_small.render(f"EN: {int(self.player.energy)}/{self.player.max_energy}", True, COLOR_UI_TEXT)
        ui_panel.blit(energy_text, (15, 31))

        # Weapon display
        weapon_name = self.player.unlocked_weapons[self.player.active_weapon_idx].replace("_", " ")
        weapon_text = self.font_small.render(f"Weapon: {weapon_name}", True, COLOR_UI_TEXT)
        ui_panel.blit(weapon_text, (10, 55))
        
        self.screen.blit(ui_panel, (10, 10))

        # Score and Steps
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, COLOR_UI_TEXT)
        self.screen.blit(score_text, (SCREEN_WIDTH - score_text.get_width() - 15, 10))
        steps_text = self.font_small.render(f"Steps: {self.steps}/{MAX_EPISODE_STEPS}", True, COLOR_UI_TEXT)
        self.screen.blit(steps_text, (SCREEN_WIDTH - steps_text.get_width() - 15, 40))

        # Minimap
        map_size = 80
        map_pos = (SCREEN_WIDTH - map_size - 10, SCREEN_HEIGHT - map_size - 10)
        map_rect = pygame.Rect(map_pos[0], map_pos[1], map_size, map_size)
        map_surface = pygame.Surface((map_size, map_size), pygame.SRCALPHA)
        map_surface.fill(COLOR_UI_BG)
        pygame.draw.rect(map_surface, COLOR_UI_TEXT, (0,0,map_size,map_size), 1)

        # Player on map
        pygame.draw.circle(map_surface, COLOR_PLAYER, (map_size//2, map_size//2), 2)
        
        # Enemies on map
        for enemy in self.enemies:
            map_x = int((enemy.x / SCREEN_WIDTH) * map_size)
            map_y = int((enemy.y / SCREEN_HEIGHT) * map_size)
            if 0 <= map_x < map_size and 0 <= map_y < map_size:
                color = COLOR_ENEMY_BOSS if enemy.is_boss else COLOR_ENEMY_SWARMER
                size = 3 if enemy.is_boss else 1
                pygame.draw.circle(map_surface, color, (map_x, map_y), size)
        
        # Salvage on map
        for item in self.salvage_items:
            map_x = int((item.x / SCREEN_WIDTH) * map_size)
            map_y = int((item.y / SCREEN_HEIGHT) * map_size)
            if 0 <= map_x < map_size and 0 <= map_y < map_size:
                pygame.draw.circle(map_surface, COLOR_SALVAGE, (map_x, map_y), 1)

        self.screen.blit(map_surface, map_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player.health,
            "energy": self.player.energy,
            "enemies": len(self.enemies),
            "boss_active": self.boss_active,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # This method is for internal validation and is not part of the standard Gym API
        try:
            # Test action space
            assert self.action_space.shape == (3,)
            assert self.action_space.nvec.tolist() == [5, 2, 2]
            
            # Test observation space  
            test_obs, _ = self.reset()
            assert test_obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
            assert test_obs.dtype == np.uint8
            
            # Test reset
            obs, info = self.reset()
            assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
            assert isinstance(info, dict)
            
            # Test step
            test_action = self.action_space.sample()
            obs, reward, term, trunc, info = self.step(test_action)
            assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
            assert isinstance(reward, (int, float))
            assert isinstance(term, bool)
            assert isinstance(trunc, bool)
            assert isinstance(info, dict)
            
            print("✓ Implementation validated successfully")
        except Exception as e:
            print(f"✗ Implementation validation failed: {e}")


if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It has been modified to use pygame.display for human interaction.
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="human_equivalent")
    obs, info = env.reset()
    
    pygame.display.set_caption("Vector Combat Arena")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    game_over = False
    
    while running:
        if game_over:
            # Display game over message
            font = pygame.font.SysFont("sans-serif", 50)
            text = font.render("GAME OVER", True, COLOR_WHITE)
            text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 - 30))
            screen.blit(text, text_rect)
            
            score_text = env.font_large.render(f"Final Score: {int(env.score)}", True, COLOR_UI_TEXT)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 20))
            screen.blit(score_text, score_rect)

            reset_text = env.font_large.render("Press 'R' to Reset", True, COLOR_UI_TEXT)
            reset_rect = reset_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 60))
            screen.blit(reset_text, reset_rect)
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    game_over = False
            continue

        # Get human input
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        game_over = terminated or truncated
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(FPS)
        
    env.close()