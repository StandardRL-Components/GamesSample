import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:49:02.981758
# Source Brief: brief_01297.md
# Brief Index: 1297
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend a crumbling space station against waves of alien attackers. "
        "Jump between platforms, repair the station, and use your stealth ability to survive."
    )
    user_guide = (
        "Controls: Use ←→ arrows to move and ↑ to jump. Press space to activate stealth. "
        "Stand in a green repair zone and press shift to repair platforms."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1000
        self.TOTAL_WAVES = 20

        # Colors
        self.COLOR_BG = (10, 20, 40)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.COLOR_STEALTH = (100, 100, 255, 128)
        self.COLOR_PLATFORM_STABLE = (100, 100, 120)
        self.COLOR_PLATFORM_CRUMBLE = (255, 180, 0)
        self.COLOR_REPAIR_ZONE = (0, 255, 0)
        self.COLOR_ENEMY_PATROLLER = (255, 100, 0)
        self.COLOR_ENEMY_SENTRY = (255, 50, 50)
        self.COLOR_ENEMY_BOMBER = (255, 0, 150)
        self.COLOR_PROJECTILE_ENEMY = (255, 0, 0)
        self.COLOR_PROJECTILE_TURRET = (0, 255, 150)
        self.COLOR_TURRET = (0, 200, 150)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BAR = (40, 60, 80)
        self.COLOR_HEALTH = (0, 255, 0)
        self.COLOR_STEALTH_COOLDOWN = (0, 150, 255)

        # Player Physics & Stats
        self.PLAYER_SIZE = 12
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_THRUST = 1.5
        self.PLAYER_GRAVITY = 0.4
        self.PLAYER_DRAG = 0.98
        self.PLAYER_MAX_HEALTH = 100
        self.STEALTH_DURATION = 90  # 3 seconds
        self.STEALTH_COOLDOWN = 180 # 6 seconds

        # Reward structure
        self.REWARD_STEP_SURVIVED = 0.01 # Changed from 0.1 to keep non-terminal rewards small
        self.REWARD_REPAIR = 5.0
        self.REWARD_WAVE_SURVIVED = 10.0
        self.REWARD_WIN = 100.0
        self.REWARD_LOSS = -100.0
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0
        self.player = {}
        self.platforms = []
        self.repair_zones = []
        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.starfield = []
        self.prev_shift_held = False

        self.reset()
        # self.validate_implementation() # Commented out for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_wave = 0
        self.prev_shift_held = False

        self.player = {
            'pos': np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float),
            'vel': np.array([0.0, 0.0], dtype=float),
            'health': self.PLAYER_MAX_HEALTH,
            'stealth_timer': 0,
            'stealth_cooldown': 0,
            'on_ground': False
        }
        
        self._initialize_level()

        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        if not self.starfield:
             for _ in range(150):
                self.starfield.append([
                    random.uniform(0, self.SCREEN_WIDTH),
                    random.uniform(0, self.SCREEN_HEIGHT),
                    random.uniform(0.1, 0.5)
                ])

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = self.REWARD_STEP_SURVIVED
        
        self._handle_input(action)
        self._update_player()
        self._update_platforms()
        self._update_enemies()
        self._update_turrets()
        self._update_projectiles()
        self._update_particles()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        wave_reward = self._check_wave_completion()
        reward += wave_reward

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.current_wave > self.TOTAL_WAVES:
                reward = self.REWARD_WIN
            else:
                reward = self.REWARD_LOSS
            self.game_over = True
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _initialize_level(self):
        self.platforms = []
        platform_data = [
            # Main floor
            (50, 350, 540, 20, 'stable'),
            # Floating platforms
            (100, 250, 100, 15, 'crumbling'),
            (250, 180, 140, 15, 'stable'),
            (450, 250, 100, 15, 'crumbling'),
            (50, 120, 80, 15, 'stable'),
            (510, 120, 80, 15, 'stable'),
        ]
        for x, y, w, h, p_type in platform_data:
            self.platforms.append({
                'rect': pygame.Rect(x, y, w, h),
                'type': p_type,
                'state': 'intact', # intact, crumbling, destroyed
                'timer': 0
            })
        
        self.repair_zones = [pygame.Rect(295, 320, 50, 30)]
        self.turrets = [{'pos': (40, 340), 'cooldown': 0, 'active': False}, {'pos': (600, 340), 'cooldown': 0, 'active': False}]

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1 and self.player['on_ground']: # Up (Jump)
            self.player['vel'][1] = -self.PLAYER_THRUST * 2.5
            # sfx: jump
        if movement == 2: # Down
            self.player['vel'][1] += self.PLAYER_ACCEL
        if movement == 3: # Left
            self.player['vel'][0] -= self.PLAYER_ACCEL
        if movement == 4: # Right
            self.player['vel'][0] += self.PLAYER_ACCEL

        if space_held and self.player['stealth_cooldown'] == 0:
            self.player['stealth_timer'] = self.STEALTH_DURATION
            self.player['stealth_cooldown'] = self.STEALTH_COOLDOWN + self.STEALTH_DURATION
            # sfx: stealth_activate

        if shift_held and not self.prev_shift_held:
            player_rect = pygame.Rect(self.player['pos'][0] - self.PLAYER_SIZE/2, self.player['pos'][1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
            for zone in self.repair_zones:
                if player_rect.colliderect(zone):
                    self.score += self._repair_platforms(zone)
                    # sfx: repair_success
        self.prev_shift_held = shift_held

    def _update_player(self):
        # Update timers
        if self.player['stealth_timer'] > 0: self.player['stealth_timer'] -= 1
        if self.player['stealth_cooldown'] > 0: self.player['stealth_cooldown'] -= 1

        # Apply physics
        self.player['vel'][1] += self.PLAYER_GRAVITY
        self.player['vel'] *= self.PLAYER_DRAG
        self.player['pos'] += self.player['vel']

        # Boundary checks
        self.player['pos'][0] = np.clip(self.player['pos'][0], 0, self.SCREEN_WIDTH)
        if self.player['pos'][0] == 0 or self.player['pos'][0] == self.SCREEN_WIDTH:
            self.player['vel'][0] = 0

        # Platform collisions
        self.player['on_ground'] = False
        player_rect = pygame.Rect(self.player['pos'][0] - self.PLAYER_SIZE/2, self.player['pos'][1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for p in self.platforms:
            if p['state'] != 'destroyed' and player_rect.colliderect(p['rect']):
                # Check collision side
                if self.player['vel'][1] > 0 and player_rect.bottom > p['rect'].top and player_rect.top < p['rect'].top: # Landing on top
                    self.player['pos'][1] = p['rect'].top - self.PLAYER_SIZE / 2
                    self.player['vel'][1] = 0
                    self.player['on_ground'] = True
                    if p['type'] == 'crumbling' and p['state'] == 'intact':
                        p['state'] = 'crumbling'
                        p['timer'] = 90 # 3 seconds
                elif self.player['vel'][1] < 0 and player_rect.top < p['rect'].bottom: # Hitting bottom
                    self.player['pos'][1] = p['rect'].bottom + self.PLAYER_SIZE / 2
                    self.player['vel'][1] = 0
                elif self.player['vel'][0] != 0 and player_rect.centery > p['rect'].top and player_rect.centery < p['rect'].bottom: # Hitting side
                    if self.player['vel'][0] > 0: self.player['pos'][0] = p['rect'].left - self.PLAYER_SIZE/2
                    else: self.player['pos'][0] = p['rect'].right + self.PLAYER_SIZE/2
                    self.player['vel'][0] = 0

    def _update_platforms(self):
        for p in self.platforms:
            if p['state'] == 'crumbling':
                p['timer'] -= 1
                if p['timer'] <= 0:
                    p['state'] = 'destroyed'
                    # sfx: platform_collapse
                    for _ in range(30):
                        self._spawn_particles(1, p['rect'].center, (150, 150, 150), 30, 3)

    def _update_enemies(self):
        for enemy in self.enemies:
            player_visible = self.player['stealth_timer'] == 0
            dist_to_player = np.linalg.norm(self.player['pos'] - enemy['pos'])

            if enemy['type'] == 'patroller':
                enemy['pos'][0] += enemy['vel'][0]
                if enemy['pos'][0] < 0 or enemy['pos'][0] > self.SCREEN_WIDTH:
                    enemy['vel'][0] *= -1
                if player_visible and dist_to_player < 300 and abs(self.player['pos'][1] - enemy['pos'][1]) < 20:
                    enemy['cooldown'] -= 1
                    if enemy['cooldown'] <= 0:
                        self._fire_projectile(enemy['pos'], self.player['pos'], self.COLOR_PROJECTILE_ENEMY, 'enemy')
                        enemy['cooldown'] = enemy['fire_rate']
                        # sfx: enemy_shoot
            
            elif enemy['type'] == 'sentry':
                 if player_visible:
                    angle_to_player = math.atan2(self.player['pos'][1] - enemy['pos'][1], self.player['pos'][0] - enemy['pos'][0])
                    if abs(angle_to_player - enemy['angle']) < enemy['cone_angle'] / 2:
                        enemy['cooldown'] -= 1
                        if enemy['cooldown'] <= 0:
                            self._fire_projectile(enemy['pos'], self.player['pos'], self.COLOR_PROJECTILE_ENEMY, 'enemy')
                            enemy['cooldown'] = enemy['fire_rate']
                            # sfx: enemy_shoot
            
            elif enemy['type'] == 'bomber':
                if enemy['state'] == 'patrolling':
                    enemy['pos'][0] += enemy['vel'][0]
                    if enemy['pos'][0] < 0 or enemy['pos'][0] > self.SCREEN_WIDTH:
                        enemy['vel'][0] *= -1
                    if player_visible and dist_to_player < 250:
                        enemy['state'] = 'diving'
                        # sfx: bomber_dive_alarm
                elif enemy['state'] == 'diving':
                    direction = (self.player['pos'] - enemy['pos'])
                    norm = np.linalg.norm(direction)
                    if norm > 1:
                        enemy['pos'] += (direction / norm) * enemy['dive_speed']
                    if dist_to_player < 20:
                        self.player['health'] -= 25
                        enemy['health'] = 0 # Explodes on contact
                        self._spawn_particles(50, enemy['pos'], self.COLOR_ENEMY_BOMBER, 40, 5)
                        # sfx: explosion

    def _update_turrets(self):
        for turret in self.turrets:
            if not turret['active']: continue
            turret['cooldown'] = max(0, turret['cooldown'] - 1)
            if turret['cooldown'] == 0:
                closest_enemy = None
                min_dist = 400
                for enemy in self.enemies:
                    dist = np.linalg.norm(enemy['pos'] - np.array(turret['pos']))
                    if dist < min_dist:
                        min_dist = dist
                        closest_enemy = enemy
                
                if closest_enemy:
                    self._fire_projectile(turret['pos'], closest_enemy['pos'], self.COLOR_PROJECTILE_TURRET, 'turret')
                    turret['cooldown'] = 45 # 1.5s
                    # sfx: turret_shoot

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            if not (0 < proj['pos'][0] < self.SCREEN_WIDTH and 0 < proj['pos'][1] < self.SCREEN_HEIGHT):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player['pos'][0] - self.PLAYER_SIZE/2, self.player['pos'][1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        for enemy in self.enemies[:]:
            if not enemy['alive']: continue
            enemy_rect = pygame.Rect(enemy['pos'][0] - enemy['size']/2, enemy['pos'][1] - enemy['size']/2, enemy['size'], enemy['size'])
            if player_rect.colliderect(enemy_rect):
                self.player['health'] -= 15
                self.player['vel'] += (self.player['pos'] - enemy['pos']) * 0.1 # Knockback
                # sfx: player_hit
        
        for proj in self.projectiles[:]:
            proj_rect = pygame.Rect(proj['pos'][0]-2, proj['pos'][1]-2, 4, 4)
            if proj['owner'] == 'enemy' and player_rect.colliderect(proj_rect):
                self.player['health'] -= 10
                self.projectiles.remove(proj)
                self._spawn_particles(10, self.player['pos'], self.COLOR_PLAYER, 20, 2)
                # sfx: player_hit
            elif proj['owner'] == 'turret':
                for enemy in self.enemies[:]:
                    if not enemy['alive']: continue
                    enemy_rect = pygame.Rect(enemy['pos'][0] - enemy['size']/2, enemy['pos'][1] - enemy['size']/2, enemy['size'], enemy['size'])
                    if enemy_rect.colliderect(proj_rect):
                        enemy['health'] -= 50
                        if proj in self.projectiles: self.projectiles.remove(proj)
                        if enemy['health'] <= 0 and enemy['alive']:
                            enemy['alive'] = False
                            self._spawn_particles(40, enemy['pos'], self.COLOR_ENEMY_PATROLLER, 30, 4)
                            # sfx: enemy_destroy
                        break
        
        # Prune dead enemies
        self.enemies = [e for e in self.enemies if e['alive']]
        return reward
    
    def _repair_platforms(self, zone_rect):
        repaired_count = 0
        for p in self.platforms:
            if p['state'] == 'destroyed':
                p['state'] = 'intact'
                repaired_count += 1
        
        # Unlock turrets
        if not self.turrets[0]['active']:
            self.turrets[0]['active'] = True
        elif not self.turrets[1]['active']:
            self.turrets[1]['active'] = True
        
        self._spawn_particles(50, zone_rect.center, self.COLOR_REPAIR_ZONE, 40, 5)
        return repaired_count * self.REWARD_REPAIR

    def _check_wave_completion(self):
        if not self.enemies and self.current_wave <= self.TOTAL_WAVES:
            self.current_wave += 1
            if self.current_wave <= self.TOTAL_WAVES:
                self._spawn_wave()
                return self.REWARD_WAVE_SURVIVED
        return 0

    def _spawn_wave(self):
        num_patrollers = 1 + self.current_wave // 2
        num_sentries = self.current_wave // 3
        num_bombers = self.current_wave // 4

        for _ in range(num_patrollers):
            self.enemies.append({
                'type': 'patroller', 'pos': np.array([random.choice([50, 590]), 220.], dtype=float),
                'vel': np.array([random.choice([-1, 1]) * (1 + self.current_wave * 0.05), 0], dtype=float),
                'health': 100, 'size': 16, 'cooldown': 120, 'fire_rate': max(30, 120 - self.current_wave * 2), 'alive': True
            })
        for _ in range(num_sentries):
             self.enemies.append({
                'type': 'sentry', 'pos': np.array([random.uniform(100, 540), 80.], dtype=float),
                'health': 150, 'size': 20, 'cooldown': 150, 'fire_rate': max(45, 150 - self.current_wave * 3), 'alive': True,
                'angle': math.pi/2, 'cone_angle': math.pi/2
            })
        for _ in range(num_bombers):
            self.enemies.append({
                'type': 'bomber', 'pos': np.array([random.uniform(100, 540), 40.], dtype=float),
                'vel': np.array([random.choice([-1, 1]), 0], dtype=float), 'dive_speed': 2 + self.current_wave * 0.02,
                'health': 50, 'size': 14, 'state': 'patrolling', 'alive': True
            })
    
    def _fire_projectile(self, start_pos, target_pos, color, owner):
        direction = np.array(target_pos) - np.array(start_pos)
        norm = np.linalg.norm(direction)
        if norm == 0: return
        velocity = (direction / norm) * 6
        self.projectiles.append({'pos': np.array(start_pos, dtype=float), 'vel': velocity, 'color': color, 'owner': owner})

    def _spawn_particles(self, num, pos, color, lifespan, speed):
        for _ in range(num):
            angle = random.uniform(0, 2 * math.pi)
            vel = np.array([math.cos(angle), math.sin(angle)]) * random.uniform(1, speed)
            self.particles.append({
                'pos': np.array(pos, dtype=float), 'vel': vel, 'color': color,
                'lifespan': random.randint(lifespan//2, lifespan)
            })

    def _check_termination(self):
        if self.player['health'] <= 0:
            return True
        if self.player['pos'][1] > self.SCREEN_HEIGHT + self.PLAYER_SIZE:
            return True
        if self.current_wave > self.TOTAL_WAVES:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        
        # Anti-softlock: if all platforms destroyed, end episode
        if not any(p['state'] != 'destroyed' for p in self.platforms):
             return True

        return False

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "health": self.player['health'],
            "station_integrity": sum(1 for p in self.platforms if p['state'] != 'destroyed') / len(self.platforms)
        }

    def _render_game(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        for star in self.starfield:
            star[0] = (star[0] - star[2]) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, (200, 200, 220), (int(star[0]), int(star[1])), int(star[2]))

        # --- Game Elements ---
        self._render_platforms()
        self._render_repair_zones()
        self._render_turrets()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_player()
        
        # --- UI ---
        self._render_ui()

    def _render_platforms(self):
        for p in self.platforms:
            if p['state'] != 'destroyed':
                color = self.COLOR_PLATFORM_STABLE if p['state'] == 'intact' else self.COLOR_PLATFORM_CRUMBLE
                if p['state'] == 'crumbling' and p['timer'] < 60 and self.steps % 10 < 5:
                    color = (255, 50, 0) # Flash red before collapsing
                pygame.draw.rect(self.screen, color, p['rect'])

    def _render_repair_zones(self):
        for zone in self.repair_zones:
            alpha = int(100 + 50 * math.sin(self.steps * 0.1))
            s = pygame.Surface(zone.size, pygame.SRCALPHA)
            pygame.draw.rect(s, (*self.COLOR_REPAIR_ZONE, alpha), s.get_rect())
            self.screen.blit(s, zone.topleft)

    def _render_turrets(self):
        for turret in self.turrets:
            if turret['active']:
                pygame.draw.circle(self.screen, self.COLOR_TURRET, (int(turret['pos'][0]), int(turret['pos'][1])), 8)
                pygame.draw.rect(self.screen, self.COLOR_TURRET, (turret['pos'][0]-4, turret['pos'][1], 8, 10))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            size = int(enemy['size'])
            if enemy['type'] == 'patroller':
                pygame.draw.rect(self.screen, self.COLOR_ENEMY_PATROLLER, (pos[0]-size//2, pos[1]-size//2, size, size))
            elif enemy['type'] == 'sentry':
                pygame.draw.polygon(self.screen, self.COLOR_ENEMY_SENTRY, [(pos[0], pos[1]-size//2), (pos[0]-size//2, pos[1]+size//2), (pos[0]+size//2, pos[1]+size//2)])
            elif enemy['type'] == 'bomber':
                pygame.draw.circle(self.screen, self.COLOR_ENEMY_BOMBER, pos, size//2)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            pygame.draw.circle(self.screen, proj['color'], pos, 3)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(max(0, 255 * (p['lifespan'] / 30)))
            color = (*p['color'], alpha)
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (2,2), 2)
            self.screen.blit(s, (int(p['pos'][0]-2), int(p['pos'][1]-2)))

    def _render_player(self):
        pos = (int(self.player['pos'][0]), int(self.player['pos'][1]))
        
        # Stealth effect
        if self.player['stealth_timer'] > 0:
            s = pygame.Surface((self.PLAYER_SIZE*2, self.PLAYER_SIZE*2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_STEALTH, (self.PLAYER_SIZE, self.PLAYER_SIZE), self.PLAYER_SIZE)
            self.screen.blit(s, (pos[0]-self.PLAYER_SIZE, pos[1]-self.PLAYER_SIZE))
        
        # Glow effect
        for i in range(4, 0, -1):
            alpha = 50 - i * 10
            s = pygame.Surface((self.PLAYER_SIZE*2, self.PLAYER_SIZE*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(s, self.PLAYER_SIZE, self.PLAYER_SIZE, self.PLAYER_SIZE + i, (*self.COLOR_PLAYER_GLOW, alpha))
            self.screen.blit(s, (pos[0]-self.PLAYER_SIZE, pos[1]-self.PLAYER_SIZE))
            
        # Player body
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, pos, self.PLAYER_SIZE // 2)

    def _render_ui(self):
        # Wave number
        wave_text = self.font_large.render(f"WAVE: {min(self.current_wave, self.TOTAL_WAVES)}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        
        # Health bar
        health_pct = max(0, self.player['health'] / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (self.SCREEN_WIDTH - 160, 15, 150, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (self.SCREEN_WIDTH - 160, 15, 150 * health_pct, 20))
        
        # Stealth cooldown
        stealth_cooldown_pct = max(0, self.player['stealth_cooldown'] / (self.STEALTH_COOLDOWN + self.STEALTH_DURATION))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (self.SCREEN_WIDTH - 160, 40, 150, 10))
        pygame.draw.rect(self.screen, self.COLOR_STEALTH_COOLDOWN, (self.SCREEN_WIDTH - 160, 40, 150 * (1-stealth_cooldown_pct), 10))

        # Station integrity
        integrity = sum(1 for p in self.platforms if p['state'] != 'destroyed') / len(self.platforms)
        integrity_text = self.font_small.render(f"STATION INTEGRITY: {integrity:.0%}", True, self.COLOR_TEXT)
        text_rect = integrity_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20))
        self.screen.blit(integrity_text, text_rect)
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # Use arrow keys for movement, space for stealth, left-shift for repair
    # Un-comment the line below to run with a display
    # os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.set_caption("Crumbling Station Defense")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    while not done:
        movement = 0 # no-op
        space = 0
        shift = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Blit the observation from the env to the display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        env.clock.tick(env.FPS)

    print(f"Game Over. Final Score: {info['score']:.2f}, Survived to Wave: {info['wave']}")
    env.close()