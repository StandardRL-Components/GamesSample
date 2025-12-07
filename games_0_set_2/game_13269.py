import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A 2D side-scrolling platformer where you restore a barren world with 'bloom' power, "
        "fighting back the forces of a 'Winter Witch'."
    )
    user_guide = (
        "Controls: ←→ to move, ↑ to jump/aim up, ↓ to aim down. "
        "Press space to shoot and shift to terraform platforms."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.WORLD_WIDTH = 3200
        self.FPS = 30
        self.MAX_STEPS = 2500

        # --- Colors ---
        self.COLOR_BG_TOP = (12, 10, 25)
        self.COLOR_BG_BOTTOM = (25, 22, 50)
        self.COLOR_BARREN = (94, 75, 53)
        self.COLOR_BLOOM = (60, 140, 70)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GLOW = (150, 255, 200)
        self.COLOR_ENEMY = (181, 107, 130)
        self.COLOR_ENEMY_PROJECTILE = (200, 150, 160)
        self.COLOR_PLAYER_PROJECTILE = (255, 100, 200)
        self.COLOR_WITCH = (220, 240, 255)
        self.COLOR_WITCH_ATTACK = (150, 200, 255)
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_UI_BAR = (50, 255, 150)
        self.COLOR_UI_BAR_BG = (50, 50, 50)
        self.COLOR_HEART = (255, 50, 50)

        # --- Physics & Gameplay ---
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 5.0
        self.PLAYER_JUMP = -14.0
        self.BLOOM_JUMP_BONUS = -4.0
        self.MAX_FALL_SPEED = 15.0
        self.PLAYER_HEALTH_MAX = 5
        self.BLOOM_POWER_MAX = 100.0
        self.BLOOM_POWER_REGEN = 0.2
        self.SHOOT_COST = 15
        self.TERRAFORM_COST = 25
        self.BOSS_TRIGGER_X = self.WORLD_WIDTH - self.WIDTH - 100
        
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
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = 0
        self.bloom_power = 0.0
        self.on_ground = False
        self.last_movement_dir = pygame.Vector2(1, 0) # For aiming
        self.last_space_held = False
        self.last_shift_held = False
        self.platforms = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.camera_offset = pygame.Vector2(0, 0)
        self.screen_shake = 0
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 0.02
        self.boss = {}
        self.reward_this_step = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_level()
        
        self.player_pos = pygame.Vector2(150, self.HEIGHT - 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.bloom_power = self.BLOOM_POWER_MAX
        
        self.on_ground = False
        self.on_ground_platform = None
        self.last_movement_dir = pygame.Vector2(1, 0)
        self.last_space_held = False
        self.last_shift_held = False
        
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.camera_offset = pygame.Vector2(0, 0)
        self.screen_shake = 0
        
        self.enemy_spawn_timer = 0
        self.enemy_spawn_rate = 0.02

        self.boss = {
            "active": False,
            "pos": pygame.Vector2(self.BOSS_TRIGGER_X + 250, self.HEIGHT - 150),
            "health": 200,
            "max_health": 200,
            "attack_timer": 0,
            "pattern": "shards",
            "phase": 1
        }
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, self.terminated, self.truncated, self._get_info()

        self.steps += 1
        self.reward_this_step = 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        self._handle_input(movement, space_held, shift_held)
        self._update_player()
        self._update_entities()
        self._update_camera()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        self.terminated = self.player_health <= 0 or (self.boss['active'] and self.boss['health'] <= 0)
        self.truncated = self.steps >= self.MAX_STEPS
        self.game_over = self.terminated or self.truncated

        if self.terminated:
            if self.player_health <= 0:
                self.reward_this_step -= 100.0
            else: # boss defeated
                self.reward_this_step += 100.0
        
        self.score += self.reward_this_step
        
        return (
            self._get_observation(),
            self.reward_this_step,
            self.terminated,
            self.truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # Horizontal Movement
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
            self.last_movement_dir.x = -1
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED
            self.last_movement_dir.x = 1
        else:
            self.player_vel.x = 0
        
        # Aiming
        if movement == 1: self.last_movement_dir.y = -1 # Aim Up
        elif movement == 2: self.last_movement_dir.y = 1 # Aim Down
        else: self.last_movement_dir.y = 0 # Aim Straight

        # Jumping
        if movement == 1 and self.on_ground:
            jump_power = self.PLAYER_JUMP
            if self.on_ground_platform and self.on_ground_platform['state'] == 'bloom':
                jump_power += self.BLOOM_JUMP_BONUS
            self.player_vel.y = jump_power
            self.on_ground = False

        # Shooting
        if space_held and not self.last_space_held and self.bloom_power >= self.SHOOT_COST:
            self.bloom_power -= self.SHOOT_COST
            proj_vel = self.last_movement_dir.normalize() * 12
            if proj_vel.length() == 0: proj_vel.x = 12 if self.last_movement_dir.x >= 0 else -12
            self.projectiles.append({
                'pos': self.player_pos.copy() + pygame.Vector2(0, -15),
                'vel': proj_vel,
                'owner': 'player',
                'radius': 6
            })
            for _ in range(10):
                p_vel = proj_vel.normalize() * random.uniform(1, 4) + pygame.Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
                self.particles.append(self._create_particle(
                    self.player_pos.copy() + pygame.Vector2(0, -15),
                    p_vel, self.COLOR_PLAYER_PROJECTILE, 4, 10))

        # Terraforming
        if shift_held and not self.last_shift_held and self.on_ground and self.bloom_power >= self.TERRAFORM_COST:
            if self.on_ground_platform and self.on_ground_platform['state'] == 'barren':
                self.bloom_power -= self.TERRAFORM_COST
                self.on_ground_platform['state'] = 'bloom'
                self.on_ground_platform['bloom_level'] = 0.0
                self.reward_this_step += 0.1
                for i in range(30):
                    angle = random.uniform(0, 2 * math.pi)
                    p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * random.uniform(1, 3)
                    self.particles.append(self._create_particle(
                        self.player_pos + pygame.Vector2(0, 5),
                        p_vel, (random.randint(100,255), random.randint(100,255), random.randint(100,255)), 5, 20))

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Move and collide
        self.player_pos.x += self.player_vel.x
        self.player_pos.y += self.player_vel.y
        
        # Keep player in world bounds
        self.player_pos.x = max(10, min(self.player_pos.x, self.WORLD_WIDTH - 10))

        # Fall out of world
        if self.player_pos.y > self.HEIGHT + 50:
            self.player_health -= 1
            self.reward_this_step -= 1.0 # Heavier penalty than normal damage
            self.player_pos.y = -50 # Respawn at top
            
            platforms_behind = [p['rect'].centerx for p in self.platforms if p['rect'].centerx < self.player_pos.x]
            if platforms_behind:
                self.player_pos.x = max(platforms_behind)
            else:
                self.player_pos.x = 150 # Fallback to start

            self.player_vel = pygame.Vector2(0, 0)
            self.screen_shake = 10

        # Platform collision
        self.on_ground = False
        self.on_ground_platform = None
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 25, 20, 30)

        for plat in self.platforms:
            if player_rect.colliderect(plat['rect']):
                # Player was above platform in previous frame
                if self.player_vel.y > 0 and (player_rect.bottom - self.player_vel.y) <= plat['rect'].top:
                    self.player_pos.y = plat['rect'].top
                    self.player_vel.y = 0
                    self.on_ground = True
                    self.on_ground_platform = plat

        # Regenerate bloom power
        self.bloom_power = min(self.BLOOM_POWER_MAX, self.bloom_power + self.BLOOM_POWER_REGEN)

    def _update_entities(self):
        # Update Platforms (visual effect)
        for p in self.platforms:
            if p['state'] == 'bloom' and p['bloom_level'] < 1.0:
                p['bloom_level'] = min(1.0, p['bloom_level'] + 0.05)
        
        # Update/Spawn Enemies
        self.enemy_spawn_timer += 1
        if self.enemy_spawn_timer > 1 / self.enemy_spawn_rate and not self.boss['active']:
            self.enemy_spawn_timer = 0
            # Spawn enemy on a platform ahead of the player
            possible_plats = [p for p in self.platforms if p['rect'].x > self.player_pos.x + self.WIDTH/2 and p['rect'].x < self.player_pos.x + self.WIDTH]
            if possible_plats:
                plat = random.choice(possible_plats)
                self.enemies.append({
                    'pos': pygame.Vector2(plat['rect'].centerx, plat['rect'].top - 15),
                    'health': 2,
                    'shoot_timer': random.randint(60, 120),
                    'anim_timer': 0
                })
        
        # Enemy difficulty scaling
        if self.steps > 0 and self.steps % 200 == 0:
            self.enemy_spawn_rate = min(0.05, self.enemy_spawn_rate + 0.005)

        # Update Enemies
        for enemy in self.enemies:
            enemy['shoot_timer'] -= 1
            enemy['anim_timer'] = (enemy['anim_timer'] + 0.1) % (2 * math.pi)
            if enemy['shoot_timer'] <= 0:
                enemy['shoot_timer'] = random.randint(90, 150)
                direction = (self.player_pos - enemy['pos']).normalize()
                self.projectiles.append({
                    'pos': enemy['pos'].copy(),
                    'vel': direction * 6,
                    'owner': 'enemy',
                    'radius': 5
                })

        # Update Projectiles
        player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 25, 20, 30)
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            p['pos'] += p['vel']
            
            if self.steps % 2 == 0:
                color = self.COLOR_PLAYER_PROJECTILE if p['owner'] == 'player' else self.COLOR_ENEMY_PROJECTILE
                self.particles.append(self._create_particle(p['pos'].copy(), -p['vel']*0.1, color, p['radius']*0.5, 8))

            if not (0 < p['pos'].x < self.WORLD_WIDTH and 0 < p['pos'].y < self.HEIGHT):
                projectiles_to_remove.append(i)
                continue

            if p['owner'] == 'player':
                enemies_to_remove = []
                for j, enemy in enumerate(self.enemies):
                    if (enemy['pos'] - p['pos']).length() < 15 + p['radius']:
                        enemy['health'] -= 1
                        if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                        if enemy['health'] <= 0:
                            self.reward_this_step += 1.0
                            enemies_to_remove.append(j)
                            for _ in range(20):
                                self.particles.append(self._create_particle(enemy['pos'], pygame.Vector2(random.uniform(-3,3), random.uniform(-3,3)), self.COLOR_ENEMY, 5, 25))
                for j in sorted(enemies_to_remove, reverse=True):
                    del self.enemies[j]

                if self.boss['active'] and (self.boss['pos'] - p['pos']).length() < 40 + p['radius']:
                    self.boss['health'] -= 1
                    if i not in projectiles_to_remove: projectiles_to_remove.append(i)
                    self.screen_shake = 5
            
            elif p['owner'] == 'enemy' or p['owner'] == 'witch':
                if (self.player_pos - p['pos']).length() < 15 + p['radius']:
                    self.player_health -= 1
                    self.reward_this_step -= 0.1
                    self.screen_shake = 10
                    if i not in projectiles_to_remove: projectiles_to_remove.append(i)
        
        for i in sorted(list(set(projectiles_to_remove)), reverse=True):
            del self.projectiles[i]

        # Update Particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.97

        # Update Boss
        self._update_boss()
        
    def _update_boss(self):
        if not self.boss['active'] and self.player_pos.x >= self.BOSS_TRIGGER_X:
            self.boss['active'] = True
            self.reward_this_step += 5.0
            self.enemies.clear()
        
        if self.boss['active']:
            self.boss['attack_timer'] -= 1

            if self.boss['phase'] == 1 and self.boss['health'] / self.boss['max_health'] <= 0.75:
                self.boss['phase'] = 2
                self.boss['pattern'] = 'slam'
                self.boss['attack_timer'] = 60
            
            if self.boss['attack_timer'] <= 0:
                if self.boss['pattern'] == 'shards':
                    self.boss['attack_timer'] = 120
                    for i in range(-2, 3):
                        angle = math.atan2(self.player_pos.y - self.boss['pos'].y, self.player_pos.x - self.boss['pos'].x)
                        angle += i * 0.2
                        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * 7
                        self.projectiles.append({
                            'pos': self.boss['pos'].copy(), 'vel': vel, 'owner': 'witch', 'radius': 8
                        })
                    if self.boss['phase'] == 2: self.boss['pattern'] = 'slam'

                elif self.boss['pattern'] == 'slam':
                    self.boss['attack_timer'] = 150
                    for i in [-1, 1]:
                        self.projectiles.append({
                            'pos': pygame.Vector2(self.boss['pos'].x, self.HEIGHT-15),
                            'vel': pygame.Vector2(i * 8, 0),
                            'owner': 'witch',
                            'radius': 15,
                            'type': 'shockwave'
                        })
                    self.screen_shake = 20
                    if self.boss['phase'] == 2: self.boss['pattern'] = 'shards'

    def _update_camera(self):
        target_x = self.player_pos.x - self.WIDTH / 2
        target_x = max(0, min(target_x, self.WORLD_WIDTH - self.WIDTH))
        if self.boss['active']:
            target_x = self.WORLD_WIDTH - self.WIDTH

        self.camera_offset.x += (target_x - self.camera_offset.x) * 0.1
        
        if self.screen_shake > 0:
            self.screen_shake -= 1
            shake_offset = pygame.Vector2(random.uniform(-self.screen_shake, self.screen_shake), 
                                          random.uniform(-self.screen_shake, self.screen_shake))
            self.camera_offset += shake_offset

    def _get_observation(self):
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))
        
        for i in range(3):
            scroll_speed = 0.1 + i * 0.1
            for j in range(-2, 10):
                x = (j * 400 - self.camera_offset.x * scroll_speed) % (self.WIDTH + 800) - 400
                y = self.HEIGHT - 150 + i * 40
                color = tuple(c * (0.3 + i*0.1) for c in self.COLOR_BARREN)
                pygame.draw.polygon(self.screen, color, [
                    (x, y + 100), (x + 200, y - 100 - i*20), (x + 400, y + 100)
                ])

        cam_x, cam_y = int(self.camera_offset.x), int(self.camera_offset.y)

        for plat in self.platforms:
            p_rect = plat['rect'].move(-cam_x, -cam_y)
            if p_rect.right < 0 or p_rect.left > self.WIDTH: continue
            if plat['state'] == 'barren':
                pygame.draw.rect(self.screen, self.COLOR_BARREN, p_rect)
            else:
                bloom_color = self.COLOR_BLOOM
                color = (
                    int(self.COLOR_BARREN[0] * (1 - plat['bloom_level']) + bloom_color[0] * plat['bloom_level']),
                    int(self.COLOR_BARREN[1] * (1 - plat['bloom_level']) + bloom_color[1] * plat['bloom_level']),
                    int(self.COLOR_BARREN[2] * (1 - plat['bloom_level']) + bloom_color[2] * plat['bloom_level'])
                )
                pygame.draw.rect(self.screen, color, p_rect)
                if plat['bloom_level'] > 0.5:
                    for i in range(3):
                        flower_x = p_rect.x + (i+1) * p_rect.width / 4
                        flower_y = p_rect.y
                        pygame.gfxdraw.filled_circle(self.screen, int(flower_x), int(flower_y), 3, (255, 255, 100))
                        pygame.gfxdraw.aacircle(self.screen, int(flower_x), int(flower_y), 3, (255, 255, 100))

        for p in self.particles:
            pos = (int(p['pos'].x - cam_x), int(p['pos'].y - cam_y))
            radius = int(p['radius'] * (p['lifespan'] / p['max_lifespan']))
            if radius > 0:
                alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
                color = (*p['color'], alpha)
                s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (radius, radius), radius)
                self.screen.blit(s, (pos[0]-radius, pos[1]-radius))

        for enemy in self.enemies:
            pos = (int(enemy['pos'].x - cam_x), int(enemy['pos'].y - cam_y))
            size = 10 + 2 * math.sin(enemy['anim_timer'])
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, pos, int(size))

        if self.boss['active']:
            pos = (int(self.boss['pos'].x - cam_x), int(self.boss['pos'].y - cam_y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 40, self.COLOR_WITCH)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 40, (255,255,255))

        for p in self.projectiles:
            pos = (int(p['pos'].x - cam_x), int(p['pos'].y - cam_y))
            color = self.COLOR_PLAYER_PROJECTILE if p['owner'] == 'player' else (self.COLOR_WITCH_ATTACK if p['owner'] == 'witch' else self.COLOR_ENEMY_PROJECTILE)
            if p.get('type') == 'shockwave':
                pygame.draw.rect(self.screen, color, (pos[0]-p['radius'], pos[1]-30, p['radius']*2, 30))
            else:
                pygame.draw.circle(self.screen, color, pos, int(p['radius']))

        player_screen_pos = (int(self.player_pos.x - cam_x), int(self.player_pos.y - cam_y - 15))
        glow_radius = 25
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_screen_pos[0]-glow_radius, player_screen_pos[1]-glow_radius+15))
        player_rect = pygame.Rect(0, 0, 20, 30)
        player_rect.center = player_screen_pos
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=5)
        
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        for i in range(self.PLAYER_HEALTH_MAX):
            pos = (20 + i * 30, 25)
            if i < self.player_health:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_HEART)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_HEART)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_UI_BAR_BG)
        
        bar_width = 150
        bar_height = 15
        bar_x, bar_y = self.WIDTH - bar_width - 15, 20
        fill_ratio = self.bloom_power / self.BLOOM_POWER_MAX
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_width * fill_ratio, bar_height), border_radius=4)
        text = self.font_small.render("BLOOM", True, self.COLOR_UI_TEXT)
        self.screen.blit(text, (bar_x - text.get_width() - 10, bar_y))
        
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 15))

        if self.boss['active']:
            bar_width = self.WIDTH - 100
            bar_height = 20
            bar_x, bar_y = 50, self.HEIGHT - 40
            fill_ratio = max(0, self.boss['health'] / self.boss['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_WITCH_ATTACK, (bar_x, bar_y, bar_width * fill_ratio, bar_height), border_radius=5)
            text = self.font_large.render("WINTER WITCH", True, self.COLOR_UI_TEXT)
            self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, bar_y - 25))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "bloom_power": self.bloom_power,
            "boss_active": self.boss['active'],
            "boss_health": self.boss['health'] if self.boss['active'] else -1,
        }

    def _generate_level(self):
        self.platforms.clear()
        self.platforms.append({'rect': pygame.Rect(50, self.HEIGHT - 80, 200, 30), 'state': 'barren', 'bloom_level': 0.0})
        x = 300
        y = self.HEIGHT - 100
        while x < self.BOSS_TRIGGER_X:
            w = random.randint(80, 150)
            self.platforms.append({'rect': pygame.Rect(x, y, w, 30), 'state': 'barren', 'bloom_level': 0.0})
            
            max_jump_x = 220
            max_jump_y = 150
            x += random.randint(80, max_jump_x)
            y += random.randint(-max_jump_y, max_jump_y)
            y = max(150, min(y, self.HEIGHT - 50))
        self.platforms.append({'rect': pygame.Rect(self.BOSS_TRIGGER_X + 100, self.HEIGHT - 80, 400, 30), 'state': 'barren', 'bloom_level': 0.0})
        self.platforms.append({'rect': pygame.Rect(0, self.HEIGHT - 5, self.WORLD_WIDTH, 10), 'state': 'barren', 'bloom_level': 0.0})
    
    def _create_particle(self, pos, vel, color, radius, lifespan):
        return {
            'pos': pos.copy(), 'vel': vel.copy(), 'color': color,
            'radius': radius, 'lifespan': lifespan, 'max_lifespan': lifespan
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("GameEnv")
    clock = pygame.time.Clock()

    total_reward = 0
    
    movement = 0
    space_held = 0
    shift_held = 0

    print(GameEnv.user_guide)
    print("R: Reset")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

        if done:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}, Info: {info}")

    env.close()