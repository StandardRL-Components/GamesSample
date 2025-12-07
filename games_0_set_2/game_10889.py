import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:09:11.952886
# Source Brief: brief_00889.md
# Brief Index: 889
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects
class Particle:
    def __init__(self, pos, vel, radius, color, lifetime):
        self.pos = list(pos)
        self.vel = list(vel)
        self.radius = radius
        self.color = color
        self.lifetime = lifetime
        self.life = float(lifetime)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.life -= 1
        return self.life > 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.lifetime))
            radius = int(self.radius * (self.life / self.lifetime))
            if radius > 0:
                # Draw a simple circle for performance
                try:
                    pygame.draw.circle(surface, self.color + (alpha,), [int(p) for p in self.pos], radius)
                except (TypeError, ValueError): # Color might not have alpha
                    pygame.draw.circle(surface, (*self.color, alpha), [int(p) for p in self.pos], radius)


class Enemy:
    def __init__(self, pos, speed):
        self.pos = list(pos)
        self.speed = speed
        self.radius = 12
        self.health = 1
        self.state_timer = random.randint(30, 90)
        self.vel = [random.choice([-1, 1]) * speed * 0.5, speed]

    def update(self):
        self.state_timer -= 1
        if self.state_timer <= 0:
            self.vel[0] *= -1 # Reverse horizontal direction
            self.state_timer = random.randint(60, 120)
        
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        # Bounce off side walls
        if self.pos[0] < self.radius or self.pos[0] > 640 - self.radius:
            self.vel[0] *= -1
            self.pos[0] = np.clip(self.pos[0], self.radius, 640 - self.radius)

    def draw(self, surface):
        p = [int(c) for c in self.pos]
        r = self.radius
        # Glow effect
        for i in range(r, 0, -2):
            alpha = int(80 * (1 - i / r))
            pygame.gfxdraw.aacircle(surface, p[0], p[1], i, (255, 50, 50, alpha))
        pygame.gfxdraw.filled_circle(surface, p[0], p[1], r-4, (255, 80, 80))
        pygame.gfxdraw.aacircle(surface, p[0], p[1], r-4, (255, 200, 200))


class Blast:
    def __init__(self, pos):
        self.pos = list(pos)
        self.speed = -15
        self.width = 80
        self.height = 10
        self.rect = pygame.Rect(pos[0] - self.width / 2, pos[1], self.width, self.height)

    def update(self):
        self.pos[1] += self.speed
        self.rect.y = self.pos[1]
        return self.rect.bottom > 0

    def draw(self, surface):
        # Draw a bright purple energy wave
        rect = self.rect.copy()
        color = (220, 100, 255)
        for i in range(5):
            alpha = 255 - i * 50
            pygame.draw.ellipse(surface, color + (alpha,), rect)
            rect.inflate_ip(i * 4, i * 2)
            rect.center = self.rect.center


class UpgradeCard:
    def __init__(self, pos):
        self.pos = list(pos)
        self.speed = 4
        self.size = 20
        self.rect = pygame.Rect(pos[0] - self.size/2, pos[1] - self.size/2, self.size, self.size)
    
    def update(self):
        self.pos[1] += self.speed
        self.rect.center = self.pos
    
    def draw(self, surface):
        # Draw a spinning green diamond
        angle = (pygame.time.get_ticks() % 1000) / 1000 * math.pi * 2
        points = []
        for i in range(4):
            a = angle + i * math.pi / 2
            points.append((
                self.pos[0] + math.cos(a) * self.size,
                self.pos[1] + math.sin(a) * self.size
            ))
        pygame.gfxdraw.aapolygon(surface, points, (100, 255, 100))
        pygame.gfxdraw.filled_polygon(surface, points, (50, 200, 50))


class Boss:
    def __init__(self):
        self.max_health = 50
        self.health = self.max_health
        self.pos = [320, -100]
        self.target_pos = [320, 100]
        self.radius = 40
        self.state = "ENTRY" # ENTRY, ATTACK_1, ATTACK_2, VULNERABLE
        self.state_timer = 120 # 2 seconds for entry
        self.projectiles = []
        self.color = (255, 0, 100)
    
    def update(self, player_pos):
        self.state_timer -= 1
        
        # Movement to target
        dx, dy = self.target_pos[0] - self.pos[0], self.target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist > 2:
            self.pos[0] += dx / dist * 3
            self.pos[1] += dy / dist * 3

        if self.state_timer <= 0:
            if self.state == "ENTRY":
                self.state = random.choice(["ATTACK_1", "ATTACK_2"])
                self.state_timer = 180 # 3 seconds attack phase
            elif self.state in ["ATTACK_1", "ATTACK_2"]:
                self.state = "VULNERABLE"
                self.state_timer = 90 # 1.5 seconds vulnerable
                self.target_pos = [random.randint(100, 540), random.randint(80, 150)]
            elif self.state == "VULNERABLE":
                self.state = random.choice(["ATTACK_1", "ATTACK_2"])
                self.state_timer = 180

        # State-specific logic
        if self.state == "ATTACK_1" and self.state_timer % 20 == 0:
            # Fire projectile at player
            # sfx: boss_fire_projectile
            p_dx, p_dy = player_pos[0] - self.pos[0], player_pos[1] - self.pos[1]
            p_dist = math.hypot(p_dx, p_dy)
            if p_dist > 0:
                vel = [p_dx / p_dist * 5, p_dy / p_dist * 5]
                self.projectiles.append(Particle(self.pos, vel, 8, self.color, 120))
        elif self.state == "ATTACK_2":
            if self.state_timer == 170: # Charge up
                self.target_pos = list(player_pos)
            if self.state_timer == 120: # Dash
                # sfx: boss_dash
                dx, dy = self.target_pos[0] - self.pos[0], self.target_pos[1] - self.pos[1]
                dist = math.hypot(dx, dy)
                if dist > 0:
                    self.pos[0] += dx/dist * 200 # Instant dash effect
                    self.pos[1] += dy/dist * 200
        
        # Update projectiles
        self.projectiles = [p for p in self.projectiles if p.update()]

    def draw(self, surface):
        p = [int(c) for c in self.pos]
        r = self.radius
        
        color = self.color if self.state != "VULNERABLE" else (255, 255, 0)
        
        # Pulsing glow
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
        glow_r = r + int(pulse * 15)
        for i in range(glow_r, r, -3):
            alpha = int(60 * (1 - (i-r)/(glow_r-r)))
            pygame.gfxdraw.aacircle(surface, p[0], p[1], i, color + (alpha,))

        pygame.gfxdraw.filled_circle(surface, p[0], p[1], r, color)
        pygame.gfxdraw.aacircle(surface, p[0], p[1], r, (255, 255, 255))
        
        for proj in self.projectiles:
            proj.draw(surface)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a futuristic highway, pilot your ship to dodge enemy fire, and unleash powerful abilities "
        "to defeat waves of foes and a final boss."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to activate your shield and shift to fire a wide energy blast."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 5000
        self.BOSS_SPAWN_STEP = 4500

        # Colors
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_SHIELD = (100, 255, 200)
        self.COLOR_HIGHWAY = (50, 20, 80)
        
        # Exact spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 20)
        self.big_font = pygame.font.SysFont("Consolas", 40, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.player_max_health = 10
        self.player_radius = 15

        self.shield_cooldown = 150
        self.shield_duration = 90
        self.shield_cooldown_timer = 0
        self.shield_active_timer = 0
        
        self.blast_cooldown = 60
        self.blast_cooldown_timer = 0
        self.blasts = []

        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.highway_lanes = []
        self.highway_scroll_speed = 4
        self.highway_center = self.WIDTH / 2
        
        self.enemies = []
        self.enemy_spawn_timer = 0
        self.difficulty = 1.0

        self.particles = []
        self.upgrade_cards = []
        self.card_spawn_timer = 0

        self.boss = None
        self.boss_spawned = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 50]
        self.player_vel = [0, 0]
        self.player_health = self.player_max_health
        
        self.shield_cooldown_timer = 0
        self.shield_active_timer = 0
        self.blast_cooldown_timer = 0
        self.blasts = []
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.highway_lanes = []
        self.highway_center = self.WIDTH / 2
        for i in range(self.HEIGHT // 20 + 1):
            self._generate_highway_lane(i * 20)
        
        self.enemies = []
        self.enemy_spawn_timer = 60
        self.difficulty = 1.0

        self.particles = []
        self.upgrade_cards = []
        self.card_spawn_timer = 300

        self.boss = None
        self.boss_spawned = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        self._handle_input(action)
        self._update_world()
        
        step_rewards = self._handle_collisions_and_rewards()
        reward += step_rewards
        
        # Constant reward for progressing
        reward += 0.01

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.player_health <= 0:
                reward += -100
            elif self.boss and self.boss.health <= 0:
                reward += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        accel = 1.2
        if movement == 1: self.player_vel[1] -= accel # Up
        if movement == 2: self.player_vel[1] += accel # Down
        if movement == 3: self.player_vel[0] -= accel # Left
        if movement == 4: self.player_vel[0] += accel # Right
        
        # Primary Ability: Shield
        if space_held and not self.prev_space_held and self.shield_cooldown_timer <= 0:
            self.shield_active_timer = self.shield_duration
            self.shield_cooldown_timer = self.shield_cooldown
            # sfx: shield_activate
            for _ in range(50):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                p = Particle(self.player_pos, vel, random.randint(2, 5), self.COLOR_SHIELD, 30)
                self.particles.append(p)
        
        # Secondary Ability: Blast
        if shift_held and not self.prev_shift_held and self.blast_cooldown_timer <= 0:
            self.blast_cooldown_timer = self.blast_cooldown
            self.blasts.append(Blast(self.player_pos))
            # sfx: blast_fire
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_world(self):
        # Update player
        self.player_vel[0] *= 0.9  # Friction
        self.player_vel[1] *= 0.9
        self.player_pos[0] += self.player_vel[0]
        self.player_pos[1] += self.player_vel[1]
        self.player_pos[0] = np.clip(self.player_pos[0], self.player_radius, self.WIDTH - self.player_radius)
        self.player_pos[1] = np.clip(self.player_pos[1], self.player_radius, self.HEIGHT - self.player_radius)

        # Update cooldowns and timers
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1
        if self.shield_active_timer > 0: self.shield_active_timer -= 1
        if self.blast_cooldown_timer > 0: self.blast_cooldown_timer -= 1
        
        # Update highway
        for lane in self.highway_lanes:
            lane['y'] += self.highway_scroll_speed
        self.highway_lanes = [lane for lane in self.highway_lanes if lane['y'] < self.HEIGHT]
        if self.highway_lanes and self.highway_lanes[-1]['y'] > 20:
            self._generate_highway_lane(self.highway_lanes[-1]['y'] - 20)

        # Update entities
        for enemy in self.enemies: enemy.update()
        for card in self.upgrade_cards: card.update()
        self.blasts = [b for b in self.blasts if b.update()]
        self.particles = [p for p in self.particles if p.update()]
        
        # Despawn off-screen entities
        self.enemies = [e for e in self.enemies if e.pos[1] < self.HEIGHT + e.radius]
        self.upgrade_cards = [c for c in self.upgrade_cards if c.pos[1] < self.HEIGHT + c.size]
        
        # Spawning logic
        self._update_spawners()
        
        # Update boss
        if self.boss:
            self.boss.update(self.player_pos)
            if self.boss.health <= 0:
                self.game_over = True
                # sfx: boss_explosion
                self._create_explosion(self.boss.pos, 300, self.boss.color, 10)


    def _update_spawners(self):
        # Update difficulty
        if self.steps % 500 == 0 and self.steps > 0:
            self.difficulty = min(3.0, self.difficulty + 0.05)

        # Spawn enemies
        if not self.boss:
            self.enemy_spawn_timer -= 1
            if self.enemy_spawn_timer <= 0:
                spawn_x = random.randint(50, self.WIDTH - 50)
                self.enemies.append(Enemy([spawn_x, -20], self.highway_scroll_speed * self.difficulty))
                self.enemy_spawn_timer = max(15, int(60 / self.difficulty))
        
        # Spawn cards
        self.card_spawn_timer -= 1
        if self.card_spawn_timer <= 0:
            spawn_x = random.randint(100, self.WIDTH - 100)
            self.upgrade_cards.append(UpgradeCard([spawn_x, -20]))
            self.card_spawn_timer = random.randint(400, 600)

        # Spawn boss
        if self.steps >= self.BOSS_SPAWN_STEP and not self.boss_spawned:
            self.boss = Boss()
            self.boss_spawned = True
            self.enemies.clear() # Clear regular enemies for the boss fight
            # sfx: boss_spawn

    def _handle_collisions_and_rewards(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.player_radius, self.player_pos[1] - self.player_radius, self.player_radius * 2, self.player_radius * 2)

        # Player vs Enemies/Projectiles
        if self.shield_active_timer <= 0:
            enemies_to_remove = []
            for enemy in self.enemies:
                if math.hypot(self.player_pos[0] - enemy.pos[0], self.player_pos[1] - enemy.pos[1]) < self.player_radius + enemy.radius:
                    self.player_health -= 1
                    reward -= 0.5
                    self._create_explosion(enemy.pos, 50, (255, 50, 50))
                    enemies_to_remove.append(enemy)
                    # sfx: player_hit
            self.enemies = [e for e in self.enemies if e not in enemies_to_remove]

            if self.boss:
                # Boss body collision
                if math.hypot(self.player_pos[0] - self.boss.pos[0], self.player_pos[1] - self.boss.pos[1]) < self.player_radius + self.boss.radius:
                    self.player_health -= 2
                    reward -= 1.0
                    # sfx: player_hit_strong
                # Boss projectiles
                projectiles_to_remove = []
                for p in self.boss.projectiles:
                    if math.hypot(self.player_pos[0] - p.pos[0], self.player_pos[1] - p.pos[1]) < self.player_radius + p.radius:
                        self.player_health -= 1
                        reward -= 0.5
                        projectiles_to_remove.append(p)
                        # sfx: player_hit
                self.boss.projectiles = [p for p in self.boss.projectiles if p not in projectiles_to_remove]
        
        # Blasts vs Enemies/Boss
        blasts_to_remove = []
        enemies_hit_by_blast = []
        for blast in self.blasts:
            for enemy in self.enemies:
                if enemy not in enemies_hit_by_blast and blast.rect.colliderect(pygame.Rect(enemy.pos[0]-enemy.radius, enemy.pos[1]-enemy.radius, enemy.radius*2, enemy.radius*2)):
                    enemies_hit_by_blast.append(enemy)
                    self.score += 10
                    reward += 1
                    self._create_explosion(enemy.pos, 50, (255, 50, 50))
                    # sfx: enemy_explode
            if self.boss and self.boss.state == "VULNERABLE":
                if blast.rect.colliderect(pygame.Rect(self.boss.pos[0]-self.boss.radius, self.boss.pos[1]-self.boss.radius, self.boss.radius*2, self.boss.radius*2)):
                    self.boss.health -= 5
                    self.score += 50
                    reward += 5
                    if blast not in blasts_to_remove:
                        blasts_to_remove.append(blast) # Blast is consumed by boss
                    # sfx: boss_hit
        self.blasts = [b for b in self.blasts if b not in blasts_to_remove]
        self.enemies = [e for e in self.enemies if e not in enemies_hit_by_blast]

        # Player vs Upgrade Cards
        cards_to_remove = []
        for card in self.upgrade_cards:
            if player_rect.colliderect(card.rect):
                self.score += 50
                reward += 5
                # Reset cooldowns as a bonus
                self.shield_cooldown_timer = 0
                self.blast_cooldown_timer = 0
                cards_to_remove.append(card)
                # sfx: collect_upgrade
        self.upgrade_cards = [c for c in self.upgrade_cards if c not in cards_to_remove]
        
        return reward

    def _check_termination(self):
        if self.game_over: return True
        if self.player_health <= 0:
            self._create_explosion(self.player_pos, 200, self.COLOR_PLAYER, 8)
            # sfx: player_explode
            return True
        if self.steps >= self.MAX_STEPS: return True
        if self.boss and self.boss.health <= 0: return True
        return False

    def _generate_highway_lane(self, y):
        self.highway_center += self.np_random.uniform(-15, 15)
        self.highway_center = np.clip(self.highway_center, self.WIDTH * 0.3, self.WIDTH * 0.7)
        width = 150 + math.sin(y * 0.01) * 50
        self.highway_lanes.append({'y': y, 'center': self.highway_center, 'width': width})

    def _create_explosion(self, pos, count, color, speed_mult=4):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            p = Particle(pos, vel, self.np_random.integers(1, 4), color, self.np_random.integers(15, 40))
            self.particles.append(p)
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render highway
        for lane in self.highway_lanes:
            y = int(lane['y'])
            perspective = y / self.HEIGHT
            width = lane['width'] * perspective
            center = lane['center']
            color_val = 50 + int(perspective * 100)
            color = (color_val, 20, color_val + 30)
            pygame.draw.line(self.screen, color, (center - width/2, y), (center + width/2, y), max(1, int(perspective * 4)))

        # Render entities
        for p in self.particles: p.draw(self.screen)
        for card in self.upgrade_cards: card.draw(self.screen)
        for enemy in self.enemies: enemy.draw(self.screen)
        for blast in self.blasts: blast.draw(self.screen)
        if self.boss: self.boss.draw(self.screen)
        
        # Render player
        p = [int(c) for c in self.player_pos]
        r = self.player_radius
        # Glow
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.01))
        glow_r = r + int(pulse * 8)
        for i in range(glow_r, r, -2):
            alpha = int(100 * (1 - (i-r)/(glow_r-r)))
            pygame.gfxdraw.aacircle(self.screen, p[0], p[1], i, self.COLOR_PLAYER + (alpha,))
        # Ship body
        points = [
            (p[0], p[1] - r),
            (p[0] - r // 2, p[1] + r // 2),
            (p[0] + r // 2, p[1] + r // 2),
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        # Render shield
        if self.shield_active_timer > 0:
            alpha_phase = self.shield_active_timer / self.shield_duration
            radius = self.player_radius + 10
            alpha = int(100 + math.sin(alpha_phase * 20) * 50)
            pygame.gfxdraw.filled_circle(self.screen, p[0], p[1], radius, self.COLOR_SHIELD + (alpha//4,))
            pygame.gfxdraw.aacircle(self.screen, p[0], p[1], radius, self.COLOR_SHIELD + (alpha,))

    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Health bar
        health_ratio = self.player_health / self.player_max_health if self.player_max_health > 0 else 0
        bar_width = 150
        pygame.draw.rect(self.screen, (100, 0, 0), (10, self.HEIGHT - 30, bar_width, 20))
        pygame.draw.rect(self.screen, (0, 200, 0), (10, self.HEIGHT - 30, bar_width * health_ratio, 20))
        pygame.draw.rect(self.screen, (255, 255, 255), (10, self.HEIGHT - 30, bar_width, 20), 1)

        # Ability Cooldowns
        icon_size = 40
        # Shield
        shield_rect = pygame.Rect(self.WIDTH - 100, self.HEIGHT - 50, icon_size, icon_size)
        pygame.draw.rect(self.screen, (50, 50, 50), shield_rect, 2)
        shield_text = self.font.render("SPC", True, (255, 255, 255))
        self.screen.blit(shield_text, shield_text.get_rect(center=shield_rect.center))
        if self.shield_cooldown_timer > 0:
            cd_ratio = self.shield_cooldown_timer / self.shield_cooldown if self.shield_cooldown > 0 else 0
            cd_h = icon_size * cd_ratio
            cd_surf = pygame.Surface((icon_size, cd_h), pygame.SRCALPHA)
            cd_surf.fill((0, 0, 0, 180))
            self.screen.blit(cd_surf, (shield_rect.x, shield_rect.y))
        
        # Blast
        blast_rect = pygame.Rect(self.WIDTH - 50, self.HEIGHT - 50, icon_size, icon_size)
        pygame.draw.rect(self.screen, (50, 50, 50), blast_rect, 2)
        blast_text = self.font.render("SFT", True, (255, 255, 255))
        self.screen.blit(blast_text, blast_text.get_rect(center=blast_rect.center))
        if self.blast_cooldown_timer > 0:
            cd_ratio = self.blast_cooldown_timer / self.blast_cooldown if self.blast_cooldown > 0 else 0
            cd_h = icon_size * cd_ratio
            cd_surf = pygame.Surface((icon_size, cd_h), pygame.SRCALPHA)
            cd_surf.fill((0, 0, 0, 180))
            self.screen.blit(cd_surf, (blast_rect.x, blast_rect.y))

        # Boss health bar
        if self.boss:
            boss_health_ratio = self.boss.health / self.boss.max_health if self.boss.max_health > 0 else 0
            bar_width = self.WIDTH - 200
            bar_x = 100
            pygame.draw.rect(self.screen, (100, 0, 0), (bar_x, 10, bar_width, 20))
            pygame.draw.rect(self.screen, self.boss.color, (bar_x, 10, bar_width * boss_health_ratio, 20))
            pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, 10, bar_width, 20), 2)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "boss_health": self.boss.health if self.boss else -1,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # Make sure to remove the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Quantum Highway")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Pygame uses a different coordinate system for blitting numpy arrays
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause before exit
            
        clock.tick(env.FPS)
        
    env.close()