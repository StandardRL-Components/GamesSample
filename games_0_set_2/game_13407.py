import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:23:41.567023
# Source Brief: brief_03407.md
# Brief Index: 3407
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
        "Defend your space station from waves of incoming enemies. Build and aim turrets, and use a cloaking device to survive."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to aim turrets. Press space to deploy a turret and shift to activate the cloak. "
        "In the upgrade menu, use ↑↓ to navigate and space to select."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    
    # Colors
    COLOR_BG = (16, 0, 32) # Dark Purple
    COLOR_PLAYER = (0, 255, 128) # Bright Green
    COLOR_ENEMY = (255, 32, 64) # Bright Red
    COLOR_SHIELD = (64, 128, 255) # Bright Blue
    COLOR_RESOURCE = (255, 224, 64) # Bright Yellow
    COLOR_TEXT = (240, 240, 240)
    COLOR_TEXT_DIM = (128, 128, 128)
    COLOR_UI_BG = (32, 16, 64, 192)

    # Game Parameters
    STATION_MAX_HEALTH = 100
    STATION_RADIUS = 30
    INITIAL_RESOURCES = 100
    MAX_WAVES = 20
    MAX_STEPS = 4500 # 2.5 minutes at 30fps
    
    TURRET_COST = 50
    TURRET_FIRE_RATE = 0.5 # seconds
    TURRET_SLOTS = 8
    
    CLOAK_DURATION = 3.0 # seconds
    CLOAK_COOLDOWN = 10.0 # seconds
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces:
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_small = pygame.font.SysFont('Consolas', 16)
        
        # State variables are initialized in reset()
        self.stars = []
        self._generate_stars()
        
        # self.reset() is called by the wrapper or user
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Game State
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = 'UPGRADE' # Start in upgrade phase
        self.wave_number = 0
        self.wave_complete = False
        self.victory = False

        # Player State
        self.station_health = self.STATION_MAX_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.aim_angle = 0.0 # Global aim angle for all turrets
        self.turret_slots = [None] * self.TURRET_SLOTS
        self.turret_positions = self._calculate_turret_positions()

        # Cloak State
        self.cloak_active = False
        self.cloak_timer = 0.0
        self.cloak_cooldown_timer = 0.0

        # Game Objects
        self.turrets = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.enemy_spawn_queue = []
        self.enemy_spawn_timer = 0.0
        
        # Action state tracking
        self.last_space_held = False
        self.last_shift_held = False

        # Upgrade System
        self.upgrade_menu_selection = 0
        self.base_turret_damage = 10
        self.base_turret_fire_rate = self.TURRET_FIRE_RATE
        self.base_cloak_duration = self.CLOAK_DURATION
        self.piercing_laser_unlocked = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0.0

        # Unpack factorized action and detect presses
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # --- STATE LOGIC ---
        if self.game_phase == 'UPGRADE':
            self._handle_upgrade_phase(movement, space_press)
        elif self.game_phase == 'DEFENSE':
            step_reward += self._handle_defense_phase(movement, space_press, shift_press)

        # --- UPDATE TIMERS ---
        dt = 1.0 / self.FPS
        self.cloak_timer = max(0, self.cloak_timer - dt)
        self.cloak_cooldown_timer = max(0, self.cloak_cooldown_timer - dt)
        if self.cloak_timer == 0:
            self.cloak_active = False

        # --- TERMINATION ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated:
            if self.victory:
                step_reward += 100 # Victory bonus
            else:
                step_reward -= 100 # Defeat penalty
        
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    # --- PHASE HANDLERS ---
    def _handle_upgrade_phase(self, movement, space_press):
        available_upgrades = self._get_available_upgrades()
        num_options = len(available_upgrades) + 1 # +1 for "Start Wave"

        if movement == 1: # Up
            self.upgrade_menu_selection = (self.upgrade_menu_selection - 1 + num_options) % num_options
        elif movement == 2: # Down
            self.upgrade_menu_selection = (self.upgrade_menu_selection + 1) % num_options

        if space_press:
            if self.upgrade_menu_selection < len(available_upgrades):
                upgrade = available_upgrades[self.upgrade_menu_selection]
                if self.resources >= upgrade['cost']:
                    self.resources -= upgrade['cost']
                    upgrade['effect']()
                    # Sound: Upgrade purchased
            else: # "Start Next Wave" selected
                self.game_phase = 'DEFENSE'
                self.wave_number += 1
                self.wave_complete = False
                self._start_next_wave()
                # Sound: Wave start
    
    def _handle_defense_phase(self, movement, space_press, shift_press):
        step_reward = 0
        dt = 1.0 / self.FPS

        # Action: Aiming
        if movement in [1, 2]: # Up/Down
            self.aim_angle += (5 if movement == 2 else -5) * dt * 4
        if movement in [3, 4]: # Left/Right
            self.aim_angle += (5 if movement == 4 else -5) * dt * 4

        # Action: Deploy Turret
        if space_press and self.resources >= self.TURRET_COST:
            try:
                slot_index = self.turret_slots.index(None)
                self.resources -= self.TURRET_COST
                new_turret = {
                    'pos': self.turret_positions[slot_index],
                    'cooldown': 0.0,
                    'damage': self.base_turret_damage,
                    'fire_rate': self.base_turret_fire_rate,
                    'type': 'piercing' if self.piercing_laser_unlocked else 'standard'
                }
                self.turrets.append(new_turret)
                self.turret_slots[slot_index] = new_turret
                # Sound: Turret placed
            except ValueError:
                pass # All slots are full

        # Action: Activate Cloak
        if shift_press and self.cloak_cooldown_timer == 0:
            self.cloak_active = True
            self.cloak_timer = self.base_cloak_duration
            self.cloak_cooldown_timer = self.CLOAK_COOLDOWN
            # Sound: Cloak activate

        # Update Game Objects
        self._update_spawner(dt)
        self._update_turrets(dt)
        self._update_enemies(dt)
        step_reward += self._update_projectiles(dt)
        self._update_particles(dt)

        # Check for wave completion
        if not self.wave_complete and not self.enemies and not self.enemy_spawn_queue:
            self.wave_complete = True
            self.game_phase = 'UPGRADE'
            step_reward += 5 # Wave survival bonus
            self.resources += 75 + self.wave_number * 5 # End of wave resource bonus
            self.upgrade_menu_selection = 0
            if self.wave_number >= self.MAX_WAVES:
                self.victory = True
                self.game_over = True
            # Sound: Wave complete

        return step_reward

    # --- UPDATE LOGIC ---
    def _update_spawner(self, dt):
        self.enemy_spawn_timer -= dt
        if self.enemy_spawn_timer <= 0 and self.enemy_spawn_queue:
            enemy_data = self.enemy_spawn_queue.pop(0)
            self.enemies.append(enemy_data['enemy'])
            self.enemy_spawn_timer = enemy_data['delay']

    def _update_turrets(self, dt):
        for turret in self.turrets:
            turret['cooldown'] = max(0, turret['cooldown'] - dt)
            if turret['cooldown'] == 0:
                self._create_projectile(turret['pos'], self.aim_angle, 'player', turret['type'], turret['damage'])
                turret['cooldown'] = turret['fire_rate']
                # Sound: Laser fire

    def _update_enemies(self, dt):
        station_pos = (self.WIDTH / 2, self.HEIGHT / 2)
        for enemy in self.enemies:
            direction_vec = (station_pos[0] - enemy['pos'][0], station_pos[1] - enemy['pos'][1])
            dist = math.hypot(*direction_vec)
            if dist > 0:
                norm_vec = (direction_vec[0] / dist, direction_vec[1] / dist)
                enemy['pos'] = (enemy['pos'][0] + norm_vec[0] * enemy['speed'] * dt,
                                enemy['pos'][1] + norm_vec[1] * enemy['speed'] * dt)
                enemy['angle'] = math.atan2(norm_vec[1], norm_vec[0])
            
            # Simple enemy firing logic (optional, for added difficulty)
            # enemy['cooldown'] = max(0, enemy['cooldown'] - dt)
            # if enemy['cooldown'] == 0:
            #     self._create_projectile(enemy['pos'], enemy['angle'], 'enemy', 'standard', 5)
            #     enemy['cooldown'] = 3.0 + self.np_random.random() * 2.0


    def _update_projectiles(self, dt):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'] = (p['pos'][0] + p['vel'][0] * dt, p['pos'][1] + p['vel'][1] * dt)
            
            hit = False
            if 0 <= p['pos'][0] < self.WIDTH and 0 <= p['pos'][1] < self.HEIGHT:
                if p['owner'] == 'player':
                    for enemy in self.enemies[:]:
                        if math.hypot(p['pos'][0] - enemy['pos'][0], p['pos'][1] - enemy['pos'][1]) < enemy['radius']:
                            enemy['health'] -= p['damage']
                            enemy['hit_timer'] = 0.1 # For flash effect
                            reward += 0.1 # Hit reward
                            hit = True
                            if enemy['health'] <= 0:
                                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 30)
                                self.enemies.remove(enemy)
                                self.resources += 15
                                reward += 1.0 # Destroy reward
                                # Sound: Explosion
                            if p['type'] != 'piercing':
                                break # Standard projectile disappears on hit
                    if not hit:
                        projectiles_to_keep.append(p)

                elif p['owner'] == 'enemy': # Not used in current design but kept for extension
                    # (Collision logic with station would go here)
                    pass
            
            if hit and p['type'] != 'piercing':
                 self._create_explosion(p['pos'], self.COLOR_PLAYER, 5)
        
        # Enemy collision with station
        station_pos = (self.WIDTH / 2, self.HEIGHT / 2)
        for enemy in self.enemies[:]:
            if math.hypot(enemy['pos'][0] - station_pos[0], enemy['pos'][1] - station_pos[1]) < self.STATION_RADIUS + enemy['radius']:
                if not self.cloak_active:
                    damage = 25 # Flat damage on collision
                    self.station_health -= damage
                    reward -= 0.5 * damage # Damage penalty
                    self._create_explosion(station_pos, self.COLOR_PLAYER, 20)
                    # Sound: Station hit
                self._create_explosion(enemy['pos'], self.COLOR_ENEMY, 30)
                self.enemies.remove(enemy)
                # Sound: Explosion
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_particles(self, dt):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] = (p['pos'][0] + p['vel'][0] * dt, p['pos'][1] + p['vel'][1] * dt)
            p['lifespan'] -= dt
            p['radius'] = max(0, p['radius'] - p['decay'] * dt)

    # --- HELPER FUNCTIONS ---
    def _start_next_wave(self):
        num_enemies = 3 + self.wave_number
        base_health = 20
        base_speed = 30
        
        for i in range(num_enemies):
            # Increase stats by 10% per wave
            health = base_health * (1.1 ** (self.wave_number - 1))
            speed = base_speed * (1.1 ** (self.wave_number - 1))
            
            # Randomize spawn location
            edge = self.np_random.integers(4)
            if edge == 0: pos = (self.np_random.uniform(0, self.WIDTH), -20)
            elif edge == 1: pos = (self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif edge == 2: pos = (-20, self.np_random.uniform(0, self.HEIGHT))
            else: pos = (self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))

            enemy = {
                'pos': pos,
                'health': health,
                'max_health': health,
                'speed': speed,
                'radius': 12,
                'hit_timer': 0,
                'angle': 0
            }
            self.enemy_spawn_queue.append({'enemy': enemy, 'delay': i * 0.5 + 1.0})

    def _create_projectile(self, pos, angle, owner, proj_type, damage):
        speed = 400 if proj_type == 'piercing' else 600
        vel = (math.cos(angle) * speed, math.sin(angle) * speed)
        color = self.COLOR_PLAYER if owner == 'player' else self.COLOR_ENEMY
        if proj_type == 'piercing':
            color = (255, 128, 255) # Magenta for piercing
        
        self.projectiles.append({'pos': pos, 'vel': vel, 'owner': owner, 'color': color, 'type': proj_type, 'damage': damage})

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(20, 100)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            radius = self.np_random.uniform(2, 5) if num_particles > 10 else self.np_random.uniform(1, 3)
            lifespan = self.np_random.uniform(0.3, 0.8)
            self.particles.append({
                'pos': pos, 'vel': vel, 'radius': radius, 'color': color, 
                'lifespan': lifespan, 'max_lifespan': lifespan, 'decay': radius / lifespan
            })

    def _calculate_turret_positions(self):
        positions = []
        for i in range(self.TURRET_SLOTS):
            angle = 2 * math.pi * i / self.TURRET_SLOTS
            radius = self.STATION_RADIUS + 15
            x = self.WIDTH / 2 + radius * math.cos(angle)
            y = self.HEIGHT / 2 + radius * math.sin(angle)
            positions.append((x, y))
        return positions

    def _get_available_upgrades(self):
        upgrades = [
            {'name': 'Turret Damage +10%', 'cost': 50, 'effect': self._upgrade_turret_damage},
            {'name': 'Turret Fire Rate +15%', 'cost': 75, 'effect': self._upgrade_turret_firerate},
            {'name': 'Cloak Duration +25%', 'cost': 60, 'effect': self._upgrade_cloak_duration},
        ]
        if self.wave_number >= 4 and not self.piercing_laser_unlocked:
            upgrades.append({'name': 'Unlock Piercing Lasers', 'cost': 200, 'effect': self._unlock_piercing_laser})
        
        return [u for u in upgrades if self.resources >= u['cost']]

    def _upgrade_turret_damage(self): self.base_turret_damage *= 1.10
    def _upgrade_turret_firerate(self): self.base_turret_fire_rate *= 0.85
    def _upgrade_cloak_duration(self): self.base_cloak_duration *= 1.25
    def _unlock_piercing_laser(self): self.piercing_laser_unlocked = True

    def _check_termination(self):
        if self.game_over: return True
        if self.station_health <= 0:
            self.game_over = True
            return True
        return False
    
    # --- GYM INTERFACE ---
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "resources": self.resources, "health": self.station_health}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    # --- RENDERING ---
    def _render_game(self):
        self._draw_stars()
        self._draw_particles()
        self._draw_station()
        if self.cloak_active: self._draw_cloak_effect()
        self._draw_turrets()
        self._draw_projectiles()
        self._draw_enemies()
        if self.game_phase == 'DEFENSE': self._draw_aim_reticle()
    
    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.station_health / self.STATION_MAX_HEALTH)
        bar_width = 200
        pygame.draw.rect(self.screen, self.COLOR_ENEMY, (self.WIDTH/2 - bar_width/2, 10, bar_width, 15))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.WIDTH/2 - bar_width/2, 10, bar_width * health_pct, 15))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.WIDTH/2 - bar_width/2, 10, bar_width, 15), 1)

        # Info Text
        self._draw_text(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", (10, 10), self.font_main, self.COLOR_TEXT)
        self._draw_text(f"RESOURCES: {int(self.resources)}", (10, 30), self.font_main, self.COLOR_RESOURCE)
        
        # Cloak UI
        cloak_ready = self.cloak_cooldown_timer == 0
        cloak_color = self.COLOR_SHIELD if cloak_ready else self.COLOR_TEXT_DIM
        self._draw_text("CLOAK READY", (self.WIDTH - 140, 10), self.font_main, cloak_color)
        if not cloak_ready:
            cooldown_pct = self.cloak_cooldown_timer / self.CLOAK_COOLDOWN
            pygame.draw.rect(self.screen, self.COLOR_TEXT_DIM, (self.WIDTH - 140, 30, 130, 5))
            pygame.draw.rect(self.screen, self.COLOR_SHIELD, (self.WIDTH - 140, 30, 130 * (1-cooldown_pct), 5))

        if self.game_phase == 'UPGRADE':
            self._draw_upgrade_menu()
        
        if self.victory:
            self._draw_text("VICTORY", (self.WIDTH/2, self.HEIGHT/2 - 30), pygame.font.SysFont('Consolas', 60, bold=True), self.COLOR_PLAYER, center=True)
        elif self.game_over and not self.victory:
            self._draw_text("GAME OVER", (self.WIDTH/2, self.HEIGHT/2 - 30), pygame.font.SysFont('Consolas', 60, bold=True), self.COLOR_ENEMY, center=True)

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT)),
                'radius': random.uniform(0.5, 1.5),
                'brightness': random.randint(50, 150)
            })

    def _draw_stars(self):
        for star in self.stars:
            c = star['brightness']
            pygame.draw.circle(self.screen, (c,c,c), star['pos'], star['radius'])

    def _draw_station(self):
        pos = (self.WIDTH / 2, self.HEIGHT / 2)
        # Glow effect
        for i in range(10, 0, -1):
            alpha = 50 - i * 5
            color = (self.COLOR_PLAYER[0], self.COLOR_PLAYER[1], self.COLOR_PLAYER[2], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.STATION_RADIUS + i, color)
        
        pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.STATION_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.STATION_RADIUS, self.COLOR_PLAYER)

    def _draw_cloak_effect(self):
        pos = (int(self.WIDTH/2), int(self.HEIGHT/2))
        radius = self.STATION_RADIUS + 25
        
        # Pulsating alpha
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 # Varies between 0 and 1
        alpha = 64 + pulse * 64
        
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*self.COLOR_SHIELD, int(alpha)))
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*self.COLOR_SHIELD, int(alpha/4)))

    def _draw_turrets(self):
        for turret in self.turrets:
            pos = (int(turret['pos'][0]), int(turret['pos'][1]))
            p1 = (pos[0] + 8 * math.cos(self.aim_angle), pos[1] + 8 * math.sin(self.aim_angle))
            p2 = (pos[0] + 8 * math.cos(self.aim_angle + 2.356), pos[1] + 8 * math.sin(self.aim_angle + 2.356))
            p3 = (pos[0] + 8 * math.cos(self.aim_angle - 2.356), pos[1] + 8 * math.sin(self.aim_angle - 2.356))
            color = (255, 128, 255) if turret['type'] == 'piercing' else self.COLOR_PLAYER
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), color)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), color)

    def _draw_projectiles(self):
        for p in self.projectiles:
            start_pos = p['pos']
            end_pos = (p['pos'][0] - p['vel'][0] * 0.05, p['pos'][1] - p['vel'][1] * 0.05)
            pygame.draw.aaline(self.screen, p['color'], start_pos, end_pos, 2)

    def _draw_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            radius = int(enemy['radius'])
            
            # Flash when hit
            color = (255, 255, 255) if enemy.get('hit_timer', 0) > 0 else self.COLOR_ENEMY
            if enemy.get('hit_timer', 0) > 0: enemy['hit_timer'] -= 1.0/self.FPS

            # Glow effect
            for i in range(5, 0, -1):
                alpha = 40 - i * 8
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + i, (*color, alpha))
            
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)

            # Health bar
            if enemy['health'] < enemy['max_health']:
                hp_pct = enemy['health'] / enemy['max_health']
                bar_w, bar_h = 20, 4
                bar_x, bar_y = pos[0] - bar_w/2, pos[1] - radius - 8
                pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_w, bar_h))
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_w * hp_pct, bar_h))

    def _draw_particles(self):
        for p in self.particles:
            alpha = p['lifespan'] / p['max_lifespan']
            color = (*p['color'], int(255 * alpha))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _draw_aim_reticle(self):
        dist = 100
        center = (self.WIDTH/2, self.HEIGHT/2)
        x = center[0] + dist * math.cos(self.aim_angle)
        y = center[1] + dist * math.sin(self.aim_angle)
        pygame.draw.aaline(self.screen, (*self.COLOR_TEXT, 100), center, (x,y))
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 5, (*self.COLOR_TEXT, 150))

    def _draw_upgrade_menu(self):
        w, h = 400, 300
        x, y = self.WIDTH/2 - w/2, self.HEIGHT/2 - h/2
        
        # Semi-transparent background
        s = pygame.Surface((w,h), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (x, y))
        pygame.draw.rect(self.screen, self.COLOR_SHIELD, (x, y, w, h), 2)
        
        self._draw_text("UPGRADE PHASE", (x + w/2, y + 20), self.font_main, self.COLOR_TEXT, center=True)
        
        available_upgrades = self._get_available_upgrades()
        options = available_upgrades + [{'name': 'START NEXT WAVE', 'cost': 0}]
        
        for i, option in enumerate(options):
            pos_y = y + 70 + i * 30
            color = self.COLOR_RESOURCE if i == self.upgrade_menu_selection else self.COLOR_TEXT
            
            self._draw_text(option['name'], (x + 20, pos_y), self.font_main, color)
            if option['cost'] > 0:
                self._draw_text(f"{option['cost']} RES", (x + w - 80, pos_y), self.font_main, color)
            
            if i == self.upgrade_menu_selection:
                pygame.draw.rect(self.screen, self.COLOR_RESOURCE, (x+10, pos_y-2, w-20, 24), 1)

    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual play and debugging
    # It will not be run by the evaluation server.
    
    # Un-dummy the video driver for local execution
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Neon Citadel Defense")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Use a persistent state for held keys
    last_keys = pygame.key.get_pressed()
    
    while not done:
        # Convert Pygame events to Gymnasium action
        movement = 0 # none
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        # In an auto-advancing environment, we step every frame
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            done = True # In a real training loop, you would reset here.
            
    # Keep the window open for a bit after game over
    end_time = pygame.time.get_ticks()
    while pygame.time.get_ticks() - end_time < 3000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        
    env.close()