import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:17:24.773652
# Source Brief: brief_00326.md
# Brief Index: 326
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Magnetoid Storm: A base-building strategy shooter Gymnasium environment.

    The player must manage resources, build defenses, and survive waves of
    cosmic storms. The goal is to survive as long as possible.

    Action Space: MultiDiscrete([5, 2, 2])
    - Component 1 (Movement): 0=None, 1=Up, 2=Down, 3=Left, 4=Right. Controls the targeting cursor.
    - Component 2 (Spacebar): 0=Released, 1=Held. Activates the magnetizing beam on asteroids or builds structures.
    - Component 3 (Shift): 0=Released, 1=Held. A press (change from 0 to 1) cycles the selected buildable item.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.

    Reward Structure:
    - +0.1 for magnetizing an asteroid.
    - +0.5 for each resource unit collected.
    - +1.0 for destroying an enemy projectile.
    - +5.0 for surviving a storm wave.
    - -100.0 (terminal) if the base is destroyed.
    - +100.0 (terminal) if the maximum episode steps are reached.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from cosmic storms by magnetizing asteroids for resources. "
        "Build automated workers and defense turrets to survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to magnetize asteroids "
        "or build structures. Press shift to cycle between building defenses and workers."
    )
    auto_advance = True

    # --- CONFIGURATION CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (15, 18, 32)
    COLOR_BASE = (0, 150, 255)
    COLOR_BASE_GLOW = (0, 75, 128)
    COLOR_ASTEROID = (120, 110, 100)
    COLOR_MAGNETIZED = (0, 255, 255)
    COLOR_WORKER = (50, 200, 255)
    COLOR_DEFENSE = (255, 200, 0)
    COLOR_STORM_PROJ = (255, 50, 50)
    COLOR_DEFENSE_PROJ = (0, 255, 150)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_ACCENT = (255, 255, 0)
    COLOR_HEALTH = (50, 255, 50)
    COLOR_HEALTH_BG = (100, 0, 0)

    # Game Mechanics
    INITIAL_RESOURCES = 50
    INITIAL_BASE_HEALTH = 1000
    BASE_RADIUS = 35
    BUILD_RADIUS = 150

    WORKER_COST = 25
    WORKER_SPEED = 2.0
    WORKER_CAPACITY = 10

    DEFENSE_COST = 40
    DEFENSE_HEALTH = 100
    DEFENSE_RANGE = 150
    DEFENSE_COOLDOWN = 60  # steps
    DEFENSE_PROJ_SPEED = 5.0

    ASTEROID_SPAWN_RATE = 0.03  # per step
    ASTEROID_MIN_RESOURCES = 15
    ASTEROID_MAX_RESOURCES = 40
    ASTEROID_SPEED_MIN = 0.5
    ASTEROID_SPEED_MAX = 1.5
    MAGNET_STRENGTH = 0.05

    WAVE_INTERVAL = 900  # steps (30 seconds)
    STORM_DURATION = 450 # steps (15 seconds)
    INITIAL_STORM_PROJ_RATE = 0.02 # per step
    INITIAL_STORM_PROJ_SPEED = 1.5

    CURSOR_SPEED = 15

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        self.base_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.buildable_items = ["DEFENSE", "WORKER"]
        self.build_costs = {"DEFENSE": self.DEFENSE_COST, "WORKER": self.WORKER_COST}
        
        # This will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.wave_timer = 0
        self.storm_active = False
        self.storm_timer = 0
        self.asteroids = []
        self.workers = []
        self.defenses = []
        self.storm_projectiles = []
        self.defense_projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(0, 0)
        self.build_selection_idx = 0
        self.prev_shift_held = False
        self.hit_flash_timer = 0
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.wave_number = 1
        self.wave_timer = self.WAVE_INTERVAL
        self.storm_active = False
        self.storm_timer = 0
        
        self.asteroids = []
        self.workers = []
        self.defenses = []
        self.storm_projectiles = []
        self.defense_projectiles = []
        self.particles = []
        
        self.cursor_pos = self.base_pos.copy()
        self.build_selection_idx = 0
        self.prev_shift_held = True # Prevent cycle on first step
        self.hit_flash_timer = 0

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        step_reward = 0

        # --- 1. Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Game Logic ---
        step_reward += self._update_world(space_held)
        self.score += step_reward

        # --- 3. Check Termination ---
        terminated = False
        truncated = False
        if self.base_health <= 0:
            terminated = True
            step_reward -= 100
            self.game_over = True
            # sfx: base_explosion
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            step_reward += 100
            self.game_over = True

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            self.cursor_pos += move_vec.normalize() * self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.SCREEN_WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.SCREEN_HEIGHT)

        # Cycle build selection on shift PRESS
        if shift_held and not self.prev_shift_held:
            self.build_selection_idx = (self.build_selection_idx + 1) % len(self.buildable_items)
            # sfx: ui_cycle
        self.prev_shift_held = shift_held

    def _update_world(self, space_held):
        reward = 0
        
        # Storm and Wave management
        if not self.storm_active:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self.storm_active = True
                self.storm_timer = self.STORM_DURATION
                # sfx: storm_warning
        else:
            self.storm_timer -= 1
            if self.storm_timer <= 0:
                self.storm_active = False
                self.wave_timer = self.WAVE_INTERVAL
                self.wave_number += 1
                reward += 5.0 # Wave survived reward
                # sfx: wave_cleared
            else:
                # Spawn storm projectiles
                storm_proj_rate = self.INITIAL_STORM_PROJ_RATE + (self.wave_number - 1) * 0.005
                if self.np_random.random() < storm_proj_rate:
                    self._spawn_storm_projectile()

        # Spawn asteroids
        if self.np_random.random() < self.ASTEROID_SPAWN_RATE and len(self.asteroids) < 20:
            self._spawn_asteroid()

        # Player actions (Magnetize/Build)
        if space_held:
            reward += self._handle_player_action()

        # Update entities
        self._update_asteroids()
        reward += self._update_workers()
        self._update_defenses()
        
        # Update and collide projectiles
        r1 = self._update_and_collide_projectiles()
        r2 = self._check_base_collisions()
        reward += r1 + r2
        
        # Update particles
        self._update_particles()
        
        # Decrement hit flash timer
        if self.hit_flash_timer > 0:
            self.hit_flash_timer -= 1
            
        return reward

    def _handle_player_action(self):
        reward = 0
        
        # Check for asteroid to magnetize
        magnetize_target = None
        for asteroid in self.asteroids:
            if not asteroid['magnetized'] and self.cursor_pos.distance_to(asteroid['pos']) < asteroid['radius']:
                magnetize_target = asteroid
                break
        
        if magnetize_target:
            magnetize_target['magnetized'] = True
            reward += 0.1
            # sfx: magnetize_start
        else: # If not magnetizing, try to build
            dist_to_base = self.cursor_pos.distance_to(self.base_pos)
            if self.BASE_RADIUS < dist_to_base < self.BUILD_RADIUS:
                build_item = self.buildable_items[self.build_selection_idx]
                cost = self.build_costs[build_item]
                if self.resources >= cost:
                    # Check for collision with existing defenses
                    can_build = True
                    for defense in self.defenses:
                        if self.cursor_pos.distance_to(defense['pos']) < 20:
                            can_build = False
                            break
                    if can_build:
                        self.resources -= cost
                        if build_item == "DEFENSE":
                            self._create_defense(self.cursor_pos.copy())
                        elif build_item == "WORKER":
                            self._create_worker()
                        # sfx: build_complete

        return reward

    def _update_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid['magnetized']:
                accel = (self.base_pos - asteroid['pos']).normalize() * self.MAGNET_STRENGTH
                asteroid['vel'] += accel
            asteroid['pos'] += asteroid['vel']
            asteroid['angle'] += asteroid['rot_speed']

    def _update_workers(self):
        reward = 0
        for worker in self.workers:
            if worker['state'] == 'idle':
                # Find nearest unclaimed asteroid
                closest_asteroid = None
                min_dist = float('inf')
                for asteroid in self.asteroids:
                    if not asteroid['claimed']:
                        dist = worker['pos'].distance_to(asteroid['pos'])
                        if dist < min_dist:
                            min_dist = dist
                            closest_asteroid = asteroid
                if closest_asteroid:
                    closest_asteroid['claimed'] = True
                    worker['target'] = closest_asteroid
                    worker['state'] = 'fetching'

            elif worker['state'] == 'fetching':
                target_pos = worker['target']['pos']
                direction = (target_pos - worker['pos'])
                if direction.length() < self.WORKER_SPEED:
                    worker['pos'] = target_pos
                    worker['state'] = 'returning'
                    # sfx: worker_collect
                else:
                    worker['pos'] += direction.normalize() * self.WORKER_SPEED

            elif worker['state'] == 'returning':
                direction = (self.base_pos - worker['pos'])
                if direction.length() < self.WORKER_SPEED + self.BASE_RADIUS:
                    # Arrived at base
                    amount = min(worker['target']['resources'], self.WORKER_CAPACITY)
                    self.resources += amount
                    reward += amount * 0.5
                    worker['target']['resources'] -= amount
                    if worker['target']['resources'] <= 0:
                        self.asteroids.remove(worker['target'])
                    else:
                        worker['target']['claimed'] = False # Release claim
                    worker['state'] = 'idle'
                    worker['pos'] = self.base_pos.copy()
                    # sfx: resource_deposit
                else:
                    worker['pos'] += direction.normalize() * self.WORKER_SPEED
        return reward

    def _update_defenses(self):
        for defense in self.defenses:
            if defense['cooldown'] > 0:
                defense['cooldown'] -= 1
            else:
                # Find target
                closest_proj = None
                min_dist = self.DEFENSE_RANGE
                for proj in self.storm_projectiles:
                    dist = defense['pos'].distance_to(proj['pos'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_proj = proj
                
                if closest_proj:
                    # Fire
                    direction = (closest_proj['pos'] - defense['pos']).normalize()
                    self._create_defense_projectile(defense['pos'], direction)
                    defense['cooldown'] = self.DEFENSE_COOLDOWN
                    defense['angle'] = math.degrees(math.atan2(-direction.y, direction.x))
                    # sfx: defense_fire

    def _update_and_collide_projectiles(self):
        reward = 0
        
        # Update positions
        for p in self.storm_projectiles: p['pos'] += p['vel']
        for p in self.defense_projectiles: p['pos'] += p['vel']

        # Collision between projectile types
        destroyed_storm_projs = set()
        destroyed_defense_projs = set()
        for i, d_proj in enumerate(self.defense_projectiles):
            for j, s_proj in enumerate(self.storm_projectiles):
                if i in destroyed_defense_projs or j in destroyed_storm_projs:
                    continue
                if d_proj['pos'].distance_to(s_proj['pos']) < 8:
                    self._create_explosion(d_proj['pos'], self.COLOR_DEFENSE_PROJ, 15)
                    destroyed_defense_projs.add(i)
                    destroyed_storm_projs.add(j)
                    reward += 1.0
                    # sfx: projectile_hit
                    break
        
        # Filter out destroyed projectiles
        self.defense_projectiles = [p for i, p in enumerate(self.defense_projectiles) if i not in destroyed_defense_projs]
        self.storm_projectiles = [p for i, p in enumerate(self.storm_projectiles) if i not in destroyed_storm_projs]
        
        # Remove off-screen projectiles
        off_screen_bounds = pygame.Rect(-50, -50, self.SCREEN_WIDTH + 100, self.SCREEN_HEIGHT + 100)
        self.defense_projectiles = [p for p in self.defense_projectiles if off_screen_bounds.collidepoint(p['pos'])]
        self.storm_projectiles = [p for p in self.storm_projectiles if off_screen_bounds.collidepoint(p['pos'])]
        
        return reward

    def _check_base_collisions(self):
        # Asteroids hitting base
        remaining_asteroids = []
        for asteroid in self.asteroids:
            if asteroid['pos'].distance_to(self.base_pos) < self.BASE_RADIUS + asteroid['radius']:
                self.resources += asteroid['resources']
                self._create_explosion(asteroid['pos'], self.COLOR_ASTEROID, 20)
                # sfx: resource_deposit
            else:
                remaining_asteroids.append(asteroid)
        self.asteroids = remaining_asteroids
        
        # Storm projectiles hitting base
        remaining_projs = []
        for proj in self.storm_projectiles:
            if proj['pos'].distance_to(self.base_pos) < self.BASE_RADIUS:
                self.base_health -= 25
                self.hit_flash_timer = 5
                self._create_explosion(proj['pos'], self.COLOR_STORM_PROJ, 25)
                # sfx: base_hit
            else:
                remaining_projs.append(proj)
        self.storm_projectiles = remaining_projs

        # Storm projectiles hitting defenses
        for defense in self.defenses:
            remaining_projs = []
            for proj in self.storm_projectiles:
                if proj['pos'].distance_to(defense['pos']) < 10:
                    defense['health'] -= 50
                    self._create_explosion(proj['pos'], self.COLOR_STORM_PROJ, 15)
                    # sfx: defense_hit
                else:
                    remaining_projs.append(proj)
            self.storm_projectiles = remaining_projs
        
        # Remove destroyed defenses
        self.defenses = [d for d in self.defenses if d['health'] > 0]
        
        return 0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95

    # --- ENTITY CREATION ---
    def _spawn_asteroid(self):
        edge = self.np_random.integers(4)
        if edge == 0: # top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -20)
        elif edge == 1: # bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20)
        elif edge == 2: # left
            pos = pygame.Vector2(-20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # right
            pos = pygame.Vector2(self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        
        angle = math.atan2(self.base_pos.y - pos.y, self.base_pos.x - pos.x)
        angle += self.np_random.uniform(-0.5, 0.5) # Add variance
        speed = self.np_random.uniform(self.ASTEROID_SPEED_MIN, self.ASTEROID_SPEED_MAX)
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        
        resources = self.np_random.integers(self.ASTEROID_MIN_RESOURCES, self.ASTEROID_MAX_RESOURCES + 1)
        radius = 8 + resources / 5
        
        self.asteroids.append({
            'pos': pos, 'vel': vel, 'radius': radius, 'resources': resources,
            'magnetized': False, 'claimed': False, 'angle': 0, 'rot_speed': self.np_random.uniform(-1, 1)
        })

    def _spawn_storm_projectile(self):
        edge = self.np_random.integers(4)
        if edge == 0: pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -10)
        elif edge == 1: pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 10)
        elif edge == 2: pos = pygame.Vector2(-10, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: pos = pygame.Vector2(self.SCREEN_WIDTH + 10, self.np_random.uniform(0, self.SCREEN_HEIGHT))

        target_point = self.base_pos + pygame.Vector2(self.np_random.uniform(-40, 40), self.np_random.uniform(-40, 40))
        direction = (target_point - pos).normalize()
        speed = self.INITIAL_STORM_PROJ_SPEED + (self.wave_number - 1) * 0.1
        
        self.storm_projectiles.append({'pos': pos, 'vel': direction * speed})
        # sfx: storm_fire

    def _create_defense(self, pos):
        self.defenses.append({
            'pos': pos, 'health': self.DEFENSE_HEALTH, 'cooldown': 0, 'angle': -90
        })

    def _create_worker(self):
        self.workers.append({
            'pos': self.base_pos.copy(), 'state': 'idle', 'target': None
        })

    def _create_defense_projectile(self, pos, direction):
        self.defense_projectiles.append({
            'pos': pos.copy(), 'vel': direction * self.DEFENSE_PROJ_SPEED
        })

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'life': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(2, 5), 'color': color
            })
    
    # --- RENDERING ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Magnet beam
        if any(a['magnetized'] for a in self.asteroids):
            pygame.draw.line(self.screen, self.COLOR_MAGNETIZED, self.base_pos, self.cursor_pos, 2)
        
        # Asteroids
        for asteroid in self.asteroids:
            points = []
            for i in range(8):
                angle = asteroid['angle'] + i * (2 * math.pi / 8)
                dist = asteroid['radius'] + self.np_random.uniform(-2, 2)
                points.append(asteroid['pos'] + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist)
            
            color = self.COLOR_MAGNETIZED if asteroid['magnetized'] else self.COLOR_ASTEROID
            pygame.gfxdraw.aapolygon(self.screen, [(int(p.x), int(p.y)) for p in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p.x), int(p.y)) for p in points], color)

        # Workers
        for worker in self.workers:
            pygame.gfxdraw.aacircle(self.screen, int(worker['pos'].x), int(worker['pos'].y), 5, self.COLOR_WORKER)
            pygame.gfxdraw.filled_circle(self.screen, int(worker['pos'].x), int(worker['pos'].y), 5, self.COLOR_WORKER)

        # Projectiles
        for p in self.storm_projectiles:
            pygame.draw.line(self.screen, self.COLOR_STORM_PROJ, p['pos'], p['pos'] - p['vel']*2, 3)
        for p in self.defense_projectiles:
            pygame.draw.line(self.screen, self.COLOR_DEFENSE_PROJ, p['pos'], p['pos'] - p['vel'], 2)

        # Defenses
        for defense in self.defenses:
            pygame.gfxdraw.aacircle(self.screen, int(defense['pos'].x), int(defense['pos'].y), 8, self.COLOR_DEFENSE)
            pygame.gfxdraw.filled_circle(self.screen, int(defense['pos'].x), int(defense['pos'].y), 8, self.COLOR_DEFENSE)
            # Turret barrel
            end_pos = defense['pos'] + pygame.Vector2(12, 0).rotate(defense['angle'])
            pygame.draw.line(self.screen, self.COLOR_DEFENSE, defense['pos'], end_pos, 4)

        # Base
        color = (255, 255, 255) if self.hit_flash_timer > 0 else self.COLOR_BASE
        for i in range(5):
            pygame.gfxdraw.aacircle(self.screen, int(self.base_pos.x), int(self.base_pos.y), self.BASE_RADIUS + i*3, self.COLOR_BASE_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.base_pos.x), int(self.base_pos.y), self.BASE_RADIUS, color)
        pygame.gfxdraw.filled_circle(self.screen, int(self.base_pos.x), int(self.base_pos.y), self.BASE_RADIUS, color)

        # Particles
        for p in self.particles:
            if p['radius'] > 1:
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Cursor
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(self.cursor_pos.y), 10, self.COLOR_UI_ACCENT)
        pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, (self.cursor_pos.x - 15, self.cursor_pos.y), (self.cursor_pos.x - 5, self.cursor_pos.y), 2)
        pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, (self.cursor_pos.x + 5, self.cursor_pos.y), (self.cursor_pos.x + 15, self.cursor_pos.y), 2)
        pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, (self.cursor_pos.x, self.cursor_pos.y - 15), (self.cursor_pos.x, self.cursor_pos.y - 5), 2)
        pygame.draw.line(self.screen, self.COLOR_UI_ACCENT, (self.cursor_pos.x, self.cursor_pos.y + 5), (self.cursor_pos.x, self.cursor_pos.y + 15), 2)

    def _render_ui(self):
        # Resources
        res_text = self.font_small.render(f"RESOURCES: {int(self.resources)}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (10, 10))

        # Base Health Bar
        health_pct = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        bar_width = 150
        bar_height = 15
        bar_x = self.SCREEN_WIDTH - bar_width - 10
        bar_y = 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH, (bar_x, bar_y, bar_width * health_pct, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Wave Info
        if self.game_over:
             msg_text = self.font_large.render("GAME OVER", True, self.COLOR_STORM_PROJ)
             self.screen.blit(msg_text, msg_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2)))
        elif self.storm_active:
            msg = f"STORM ACTIVE | WAVE {self.wave_number}"
            msg_text = self.font_large.render(msg, True, self.COLOR_STORM_PROJ)
            self.screen.blit(msg_text, msg_text.get_rect(centerx=self.SCREEN_WIDTH/2, y=10))
        else:
            time_left = int(self.wave_timer / self.FPS)
            msg = f"WAVE {self.wave_number} | NEXT STORM IN: {time_left}"
            msg_text = self.font_small.render(msg, True, self.COLOR_TEXT)
            self.screen.blit(msg_text, msg_text.get_rect(centerx=self.SCREEN_WIDTH/2, y=10))
        
        # Build Selection
        build_item = self.buildable_items[self.build_selection_idx]
        cost = self.build_costs[build_item]
        build_text = self.font_small.render(f"BUILD: {build_item} ({cost})", True, self.COLOR_TEXT)
        self.screen.blit(build_text, build_text.get_rect(centerx=self.SCREEN_WIDTH/2, bottom=self.SCREEN_HEIGHT-10))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "resources": self.resources}
    
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
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # Arrows: Move cursor
    # Space: Magnetize/Build
    # Left Shift: Cycle build item
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = env.action_space.sample() # Default to random action
        action[0], action[1], action[2] = 0, 0, 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Actions
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a window if it doesn't exist
        if 'window' not in locals():
            pygame.display.quit() # Quit the dummy display
            pygame.display.init() # Re-init for a real display
            window = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
            pygame.display.set_caption("Magnetoid Storm")
        
        window.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Survived {info['wave'] - 1} waves.")
            running = False
            pygame.time.wait(3000) # Wait 3 seconds before closing
            
    env.close()