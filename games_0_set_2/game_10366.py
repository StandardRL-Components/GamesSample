import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:21:24.727033
# Source Brief: brief_00366.md
# Brief Index: 366
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Infiltrate a high-security arcade. Evade cameras, fight off robotic guards, and reach the high-score board to win."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press Shift to shrink and sneak past cameras. "
        "Press Space to attack enemies or interact with objects."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_EPISODE_STEPS = 2000

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 100, 200)
    COLOR_ENEMY_PATROLLER = (255, 50, 50)
    COLOR_ENEMY_SENTRY = (255, 100, 20)
    COLOR_ENEMY_GLOW = (200, 50, 50)
    COLOR_CAMERA_CONE = (255, 255, 0, 50)
    COLOR_CAMERA_BODY = (150, 150, 160)
    COLOR_TILES = (255, 200, 0)
    COLOR_TILES_SELECTED = (100, 255, 100)
    COLOR_HIGHSCORE = (255, 255, 100)
    COLOR_TEXT = (255, 255, 255)
    COLOR_HEALTH_BAR_BG = (50, 50, 50)
    COLOR_HEALTH_BAR_FG = (50, 200, 50)
    
    # Player
    PLAYER_BASE_SPEED = 4.0
    PLAYER_RADIUS_NORMAL = 12
    PLAYER_RADIUS_SHRUNK = 6
    PLAYER_ATTACK_COOLDOWN = 15 # steps
    PLAYER_ATTACK_RANGE = 40
    PLAYER_ATTACK_DAMAGE = 25

    # Enemies
    ENEMY_SPEED_INCREASE_INTERVAL = 200
    ENEMY_SPEED_INCREASE_AMOUNT = 0.05
    ENEMY_DETECTION_RADIUS = 80
    
    # Power-ups
    POWERUP_DURATION = 300 # steps (10 seconds)
    POWERUP_SPEED_BOOST = 1.5
    POWERUP_ATTACK_BOOST = 2.0
    
    # Boss
    BOSS_SPAWN_MATCH_COUNT = 3
    BOSS_HEALTH = 300
    BOSS_RADIUS = 30
    BOSS_PROJECTILE_SPEED = 5
    BOSS_SHOOT_COOLDOWN = 60 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 14, bold=True)
        
        self.render_mode = render_mode
        self._initialize_state_variables()
        # self.validate_implementation() # Commented out for submission

    def _initialize_state_variables(self):
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.terminal_reason = ""
        self.successful_matches = 0

        # Player
        self.player_pos = pygame.Vector2(0, 0)
        self.is_shrunk = False
        self.player_attack_timer = 0
        
        # Input state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Entities
        self.enemies = []
        self.cameras = []
        self.particles = []
        self.projectiles = []

        # Tile Grid
        self.tile_grid_rect = pygame.Rect(0, 0, 0, 0)
        self.selected_tiles = []
        
        # High Score
        self.highscore_rect = pygame.Rect(0, 0, 0, 0)

        # Power-ups
        self.powerups = {
            "speed_boost": 0,
            "invincibility": 0,
            "attack_boost": 0,
        }

        # Boss
        self.boss = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state_variables()
        
        self.player_pos = pygame.Vector2(self.WIDTH * 0.1, self.HEIGHT / 2)
        
        self.tile_grid_rect = pygame.Rect(self.WIDTH / 2 - 60, self.HEIGHT / 2 - 60, 120, 120)
        self.highscore_rect = pygame.Rect(self.WIDTH - 60, self.HEIGHT / 2 - 40, 40, 80)

        self._spawn_initial_entities()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self.steps += 1
        reward = 0

        # --- Update Game Logic ---
        self._update_player(movement, shift_press)
        reward += self._handle_interactions(space_press)
        
        self._update_enemies()
        self._update_cameras()
        self._update_projectiles()
        self._update_powerups()
        self._update_particles()
        if self.boss:
            self._update_boss()

        # --- Calculate Rewards & Check Termination ---
        reward_feedback, terminated_feedback = self._calculate_feedback_and_termination()
        reward += reward_feedback
        
        self.score += reward
        
        terminated = self.game_over or terminated_feedback or self.steps >= self.MAX_EPISODE_STEPS
        if self.steps >= self.MAX_EPISODE_STEPS and not self.game_over:
            self.terminal_reason = "Time Limit Reached"
        
        truncated = self.steps >= self.MAX_EPISODE_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_initial_entities(self):
        self.enemies.append(self._create_enemy('patroller', pygame.Vector2(self.WIDTH * 0.7, self.HEIGHT * 0.2)))
        self.cameras.append(self._create_camera(pygame.Vector2(self.WIDTH * 0.3, 10), -45, 45))
        self.cameras.append(self._create_camera(pygame.Vector2(self.WIDTH * 0.8, self.HEIGHT - 10), 135, 225))

    def _update_player(self, movement, shift_press):
        if shift_press:
            self.is_shrunk = not self.is_shrunk
            # sfx: shrink/grow sound

        speed_multiplier = self.POWERUP_SPEED_BOOST if self.powerups["speed_boost"] > 0 else 1.0
        current_speed = self.PLAYER_BASE_SPEED * speed_multiplier

        if movement == 1: self.player_pos.y -= current_speed
        elif movement == 2: self.player_pos.y += current_speed
        elif movement == 3: self.player_pos.x -= current_speed
        elif movement == 4: self.player_pos.x += current_speed

        player_radius = self.PLAYER_RADIUS_SHRUNK if self.is_shrunk else self.PLAYER_RADIUS_NORMAL
        self.player_pos.x = np.clip(self.player_pos.x, player_radius, self.WIDTH - player_radius)
        self.player_pos.y = np.clip(self.player_pos.y, player_radius, self.HEIGHT - player_radius)

        if self.player_attack_timer > 0:
            self.player_attack_timer -= 1

    def _handle_interactions(self, space_press):
        if not space_press:
            return 0
        
        reward = 0
        
        # 1. Highscore Board Interaction (Win)
        if self.highscore_rect.collidepoint(self.player_pos):
            self.game_over = True
            self.terminal_reason = "High Score Reached!"
            # sfx: win sound
            return 100

        # 2. Attack
        if self.player_attack_timer == 0 and not self.is_shrunk:
            attacked = False
            attack_damage = self.PLAYER_ATTACK_DAMAGE * (self.POWERUP_ATTACK_BOOST if self.powerups["attack_boost"] > 0 else 1.0)
            
            # Attack Boss
            if self.boss and self.player_pos.distance_to(self.boss['pos']) < self.PLAYER_ATTACK_RANGE + self.boss['radius']:
                self.boss['health'] -= attack_damage
                self._create_hit_particles(self.boss['pos'])
                attacked = True
                if self.boss['health'] <= 0:
                    reward += 50
                    self._create_explosion(self.boss['pos'], 50)
                    self.boss = None
                    # sfx: boss explosion
            
            # Attack Enemies
            if not attacked:
                for enemy in self.enemies:
                    if self.player_pos.distance_to(enemy['pos']) < self.PLAYER_ATTACK_RANGE + enemy['radius']:
                        enemy['health'] -= attack_damage
                        self._create_hit_particles(enemy['pos'])
                        attacked = True
                        break # Attack one at a time
            
            if attacked:
                self.player_attack_timer = self.PLAYER_ATTACK_COOLDOWN
                # sfx: player attack
                self._create_attack_swing_particles()

        # 3. Tile Grid Interaction
        player_radius = self.PLAYER_RADIUS_SHRUNK if self.is_shrunk else self.PLAYER_RADIUS_NORMAL
        if self.tile_grid_rect.inflate(player_radius*2, player_radius*2).collidepoint(self.player_pos):
            tile_size = self.tile_grid_rect.width / 3
            local_x = self.player_pos.x - self.tile_grid_rect.left
            local_y = self.player_pos.y - self.tile_grid_rect.top
            tile_idx = (int(local_x / tile_size), int(local_y / tile_size))
            
            if tile_idx not in self.selected_tiles:
                self.selected_tiles.append(tile_idx)
                # sfx: tile select
                if len(self.selected_tiles) >= 3:
                    reward += 5
                    self.successful_matches += 1
                    self.selected_tiles = []
                    self._grant_powerup()
                    self._spawn_on_match()
                    # sfx: match success
        
        return reward

    def _grant_powerup(self):
        powerup_choice = self.np_random.choice(["speed_boost", "invincibility", "attack_boost"])
        self.powerups[powerup_choice] = self.POWERUP_DURATION
        self._create_powerup_particles(self.player_pos, powerup_choice)

    def _spawn_on_match(self):
        # Spawn a new enemy
        spawn_pos = pygame.Vector2(self.np_random.uniform(0.6, 0.9) * self.WIDTH, self.np_random.uniform(0.1, 0.9) * self.HEIGHT)
        enemy_type = self.np_random.choice(['patroller', 'sentry'])
        self.enemies.append(self._create_enemy(enemy_type, spawn_pos))

        # Check for boss spawn
        if self.successful_matches >= self.BOSS_SPAWN_MATCH_COUNT and self.boss is None:
            self.boss = {
                'pos': pygame.Vector2(self.WIDTH * 0.8, self.HEIGHT / 2),
                'health': self.BOSS_HEALTH,
                'max_health': self.BOSS_HEALTH,
                'radius': self.BOSS_RADIUS,
                'shoot_timer': self.BOSS_SHOOT_COOLDOWN,
            }
            # sfx: boss spawn
    
    def _update_enemies(self):
        speed_increase = (self.steps // self.ENEMY_SPEED_INCREASE_INTERVAL) * self.ENEMY_SPEED_INCREASE_AMOUNT
        
        for enemy in self.enemies[:]:
            enemy['speed'] += speed_increase
            
            if enemy['type'] == 'patroller':
                if enemy['pos'].distance_to(enemy['target']) < enemy['speed']:
                    enemy['target'], enemy['start'] = enemy['start'], enemy['target']
                direction = (enemy['target'] - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed']
            
            elif enemy['type'] == 'sentry':
                enemy['angle'] = (enemy['angle'] + enemy['rot_speed']) % 360
            
            if enemy['health'] <= 0:
                self.score += 10 # Direct score addition for kills
                self.enemies.remove(enemy)
                self._create_explosion(enemy['pos'], 20)
                # sfx: enemy explosion
                continue
            
            # Collision with player
            player_radius = self.PLAYER_RADIUS_SHRUNK if self.is_shrunk else self.PLAYER_RADIUS_NORMAL
            if not self.powerups['invincibility'] > 0 and enemy['pos'].distance_to(self.player_pos) < enemy['radius'] + player_radius:
                self.game_over = True
                self.terminal_reason = "Caught by an Enemy"
                self.score -= 100 # Terminal penalty
                # sfx: player death

    def _update_boss(self):
        if not self.boss: return
        
        # Boss attacks
        self.boss['shoot_timer'] -= 1
        if self.boss['shoot_timer'] <= 0:
            self.boss['shoot_timer'] = self.BOSS_SHOOT_COOLDOWN
            direction = (self.player_pos - self.boss['pos']).normalize()
            self.projectiles.append({
                'pos': self.boss['pos'].copy(),
                'vel': direction * self.BOSS_PROJECTILE_SPEED,
                'radius': 8,
                'color': self.COLOR_ENEMY_PATROLLER,
            })
            # sfx: boss shoot
        
        # Collision with player
        player_radius = self.PLAYER_RADIUS_SHRUNK if self.is_shrunk else self.PLAYER_RADIUS_NORMAL
        if not self.powerups['invincibility'] > 0 and self.boss['pos'].distance_to(self.player_pos) < self.boss['radius'] + player_radius:
            self.game_over = True
            self.terminal_reason = "Defeated by Boss"
            self.score -= 100
            # sfx: player death

    def _update_cameras(self):
        for cam in self.cameras:
            cam['angle'] += cam['rot_speed']
            if not cam['min_angle'] <= cam['angle'] <= cam['max_angle']:
                cam['rot_speed'] *= -1
                cam['angle'] = np.clip(cam['angle'], cam['min_angle'], cam['max_angle'])

    def _update_projectiles(self):
        player_radius = self.PLAYER_RADIUS_SHRUNK if self.is_shrunk else self.PLAYER_RADIUS_NORMAL
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            if not (0 < p['pos'].x < self.WIDTH and 0 < p['pos'].y < self.HEIGHT):
                self.projectiles.remove(p)
                continue
            
            if not self.powerups['invincibility'] > 0 and p['pos'].distance_to(self.player_pos) < p['radius'] + player_radius:
                self.game_over = True
                self.terminal_reason = "Hit by Projectile"
                self.score -= 100
                self.projectiles.remove(p)
                # sfx: player death
    
    def _update_powerups(self):
        for p_name in self.powerups:
            if self.powerups[p_name] > 0:
                self.powerups[p_name] -= 1

    def _calculate_feedback_and_termination(self):
        reward = 0
        terminated = False
        
        # Camera detection
        is_detected = False
        if not self.is_shrunk and self.powerups["invincibility"] == 0:
            for cam in self.cameras:
                dist = self.player_pos.distance_to(cam['pos'])
                if dist < cam['range']:
                    angle_to_player = math.degrees(math.atan2(self.player_pos.y - cam['pos'].y, self.player_pos.x - cam['pos'].x))
                    angle_diff = (cam['angle'] - angle_to_player + 180) % 360 - 180
                    if abs(angle_diff) < cam['arc'] / 2:
                        is_detected = True
                        break
        
        if is_detected:
            terminated = True
            self.game_over = True
            self.terminal_reason = "Spotted by Camera"
            reward -= 100
            # sfx: camera alarm
        else:
            reward += 0.01 # Small reward for surviving
        
        # Enemy proximity penalty
        for enemy in self.enemies:
            if self.player_pos.distance_to(enemy['pos']) < self.ENEMY_DETECTION_RADIUS:
                reward -= 0.01
        if self.boss and self.player_pos.distance_to(self.boss['pos']) < self.ENEMY_DETECTION_RADIUS * 2:
             reward -= 0.02
        
        return reward, terminated

    # --- Entity Creation ---
    def _create_enemy(self, type, pos):
        if type == 'patroller':
            return {
                'type': 'patroller', 'pos': pos, 'radius': 10,
                'start': pos - pygame.Vector2(80, 0), 'target': pos + pygame.Vector2(80, 0),
                'speed': self.np_random.uniform(1.0, 1.5), 'health': 50, 'max_health': 50
            }
        else: # sentry
            return {
                'type': 'sentry', 'pos': pos, 'radius': 14,
                'angle': 0, 'rot_speed': self.np_random.uniform(1.0, 2.0) * self.np_random.choice([-1, 1]),
                'speed': 0, 'health': 80, 'max_health': 80
            }

    def _create_camera(self, pos, angle_min, angle_max):
        return {
            'pos': pos, 'angle': angle_min, 'rot_speed': self.np_random.uniform(0.5, 1.0),
            'min_angle': angle_min, 'max_angle': angle_max,
            'arc': 60, 'range': 200
        }

    # --- Particle System ---
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 31),
                'color': self.np_random.choice([self.COLOR_ENEMY_PATROLLER, self.COLOR_ENEMY_SENTRY, (255,255,255)]),
                'radius': self.np_random.uniform(1, 4)
            })

    def _create_hit_particles(self, pos):
        for _ in range(5):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(5, 11),
                'color': (200, 200, 200),
                'radius': self.np_random.uniform(1, 3)
            })

    def _create_attack_swing_particles(self):
        player_dir = pygame.Vector2(1,0) # Default right
        # Find closest enemy to determine swing direction
        closest_dist = float('inf')
        entities = self.enemies + ([self.boss] if self.boss else [])
        if entities:
            closest_entity = min(entities, key=lambda e: self.player_pos.distance_to(e['pos']))
            if self.player_pos.distance_to(closest_entity['pos']) < self.PLAYER_ATTACK_RANGE * 1.5:
                 player_dir = (closest_entity['pos'] - self.player_pos).normalize()

        for i in range(10):
            angle_offset = self.np_random.uniform(-math.pi/4, math.pi/4)
            angle = math.atan2(player_dir.y, player_dir.x) + angle_offset
            speed = 2 + i * 0.3
            self.particles.append({
                'pos': self.player_pos.copy(),
                'vel': pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': 8, 'color': (200, 200, 255), 'radius': 2
            })
            
    def _create_powerup_particles(self, pos, p_type):
        p_color = (0, 255, 0)
        if p_type == "speed_boost": p_color = (0, 200, 255)
        elif p_type == "attack_boost": p_color = (255, 100, 0)

        for i in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_y = self.np_random.uniform(-3, -1)
            vel_x = math.cos(angle) * self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(vel_x, vel_y),
                'life': self.np_random.integers(20, 41),
                'color': p_color, 'radius': self.np_random.uniform(1, 3)
            })

    # --- Rendering ---
    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "terminal_reason": self.terminal_reason}

    def _render_all(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_details()
        self._render_cameras()
        self._render_highscore()
        self._render_tile_grid()
        self._render_particles()
        self._render_projectiles()
        self._render_enemies()
        if self.boss: self._render_boss()
        self._render_player()
        self._render_ui()

    def _render_background_details(self):
        # Simulates the inside of an arcade cabinet
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, (20, 15, 35), (i, 0), (i, self.HEIGHT), 1)
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, (20, 15, 35), (0, i), (self.WIDTH, i), 1)

    def _render_player(self):
        radius = self.PLAYER_RADIUS_SHRUNK if self.is_shrunk else self.PLAYER_RADIUS_NORMAL
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow effect
        glow_radius = int(radius * 1.8)
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        glow_color = self.COLOR_PLAYER if self.powerups['invincibility'] > 0 else self.COLOR_PLAYER_GLOW
        alpha = 150 if self.powerups['invincibility'] == 0 else 255 - (self.powerups['invincibility'] % 30) * 6
        pygame.draw.circle(glow_surface, (*glow_color, alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surface, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, self.COLOR_PLAYER)
        
        # Shrink indicator
        icon_pos = (pos_int[0] + radius, pos_int[1] - radius)
        if self.is_shrunk:
            pygame.draw.line(self.screen, self.COLOR_TEXT, (icon_pos[0]-3, icon_pos[1]), (icon_pos[0]+3, icon_pos[1]), 2)
        else:
            pygame.draw.line(self.screen, self.COLOR_TEXT, (icon_pos[0]-3, icon_pos[1]), (icon_pos[0]+3, icon_pos[1]), 2)
            pygame.draw.line(self.screen, self.COLOR_TEXT, (icon_pos[0], icon_pos[1]-3), (icon_pos[0], icon_pos[1]+3), 2)


    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'].x), int(enemy['pos'].y))
            color = self.COLOR_ENEMY_PATROLLER if enemy['type'] == 'patroller' else self.COLOR_ENEMY_SENTRY
            
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(enemy['radius'] * 1.5), (*self.COLOR_ENEMY_GLOW, 100))
            
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], enemy['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], enemy['radius'], color)
            
            # Health bar
            self._render_health_bar(enemy['pos'], enemy['health'], enemy['max_health'], 30)

    def _render_boss(self):
        pos_int = (int(self.boss['pos'].x), int(self.boss['pos'].y))
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(self.boss['radius'] * 1.5), (*self.COLOR_ENEMY_GLOW, 150))
        
        # Body
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.boss['radius'], self.COLOR_ENEMY_PATROLLER)
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.boss['radius'], self.COLOR_ENEMY_PATROLLER)
        
        # Health bar
        self._render_health_bar(self.boss['pos'], self.boss['health'], self.boss['max_health'], 60)

    def _render_cameras(self):
        for cam in self.cameras:
            # Body
            pygame.gfxdraw.filled_circle(self.screen, int(cam['pos'].x), int(cam['pos'].y), 8, self.COLOR_CAMERA_BODY)
            pygame.gfxdraw.aacircle(self.screen, int(cam['pos'].x), int(cam['pos'].y), 8, self.COLOR_CAMERA_BODY)
            
            # Vision cone
            points = [cam['pos']]
            for i in range(-cam['arc'] // 2, cam['arc'] // 2 + 1, 5):
                angle = math.radians(cam['angle'] + i)
                points.append((
                    cam['pos'].x + cam['range'] * math.cos(angle),
                    cam['pos'].y + cam['range'] * math.sin(angle)
                ))
            if len(points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CAMERA_CONE)

    def _render_tile_grid(self):
        tile_size = self.tile_grid_rect.width / 3
        for r in range(3):
            for c in range(3):
                tile_rect = pygame.Rect(
                    self.tile_grid_rect.left + c * tile_size,
                    self.tile_grid_rect.top + r * tile_size,
                    tile_size, tile_size
                )
                color = self.COLOR_TILES_SELECTED if (c,r) in self.selected_tiles else self.COLOR_TILES
                pygame.draw.rect(self.screen, color, tile_rect.inflate(-4, -4), 2, border_radius=4)
    
    def _render_highscore(self):
        pygame.draw.rect(self.screen, self.COLOR_HIGHSCORE, self.highscore_rect, 0, border_radius=5)
        text = self.font_small.render("S", True, self.COLOR_BG)
        self.screen.blit(text, text.get_rect(center=self.highscore_rect.center).move(0,-20))
        text = self.font_small.render("C", True, self.COLOR_BG)
        self.screen.blit(text, text.get_rect(center=self.highscore_rect.center).move(0,-5))
        text = self.font_small.render("R", True, self.COLOR_BG)
        self.screen.blit(text, text.get_rect(center=self.highscore_rect.center).move(0,10))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 15))))
            color = (*p['color'][:3], alpha)
            if len(color) == 4:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos_int = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], p['radius'], p['color'])
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], p['radius'], p['color'])

    def _render_health_bar(self, pos, current, max_val, width):
        if current < max_val:
            bar_rect_bg = pygame.Rect(pos.x - width/2, pos.y - 30, width, 5)
            health_perc = max(0, current / max_val)
            bar_rect_fg = pygame.Rect(pos.x - width/2, pos.y - 30, width * health_perc, 5)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bar_rect_bg, 0, 2)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, bar_rect_fg, 0, 2)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Power-ups
        y_offset = 10
        powerup_map = {
            "speed_boost": ("SPD", (0, 200, 255)),
            "invincibility": ("INV", (255, 255, 255)),
            "attack_boost": ("ATK", (255, 100, 0)),
        }
        for p_name, (text, color) in powerup_map.items():
            if self.powerups[p_name] > 0:
                p_text = self.font_small.render(f"{text}: {self.powerups[p_name] / self.FPS:.1f}s", True, color)
                self.screen.blit(p_text, (10, y_offset))
                y_offset += 20
        
        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_main.render("GAME OVER", True, self.COLOR_ENEMY_PATROLLER)
            reason_text = self.font_small.render(self.terminal_reason, True, self.COLOR_TEXT)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 - 20)))
            self.screen.blit(reason_text, reason_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2 + 20)))

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # For this to work, you must comment out the line:
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    # And have pygame installed.
    
    # To re-enable headless mode for the test suite, uncomment the line.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # The following line needs to be commented out for the test suite to run
    # os.environ.pop("SDL_VIDEODRIVER", None)
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Arcade Heist")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
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
        if terminated or truncated:
            print(f"Game Over. Reason: {info.get('terminal_reason', 'Time limit')}. Score: {info.get('score')}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        clock.tick(GameEnv.FPS)

    env.close()