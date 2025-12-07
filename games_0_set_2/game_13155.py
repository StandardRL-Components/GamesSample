import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete, Box
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "You are a zero-gravity engineer repairing a damaged space station. "
        "Collect parts and fix critical systems before they fail."
    )
    user_guide = (
        "Use arrow keys to move in zero-G. Hold Shift to use the jetpack for a boost. "
        "Press Space to interact with systems, collect parts, or activate upgrades."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_STARS = (150, 150, 170)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 150, 255, 50)
    COLOR_JETPACK = (255, 255, 255)
    COLOR_WALL = (50, 60, 80)
    
    COLOR_SYS_HEALTHY = (0, 255, 100)
    COLOR_SYS_WARN = (255, 200, 0)
    COLOR_SYS_CRIT = (255, 50, 50)
    
    COLOR_RESOURCE = (255, 255, 255)
    COLOR_UPGRADE = (200, 100, 255)
    
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 40, 60, 180)

    # Player Physics
    PLAYER_SIZE = 10
    PLAYER_ACCEL = 0.8
    PLAYER_JETPACK_ACCEL = 2.0
    PLAYER_DRAG = 0.95
    MAX_VEL = 15

    # Game Mechanics
    JETPACK_FUEL_MAX = 100
    JETPACK_CONSUMPTION = 2.5
    JETPACK_REGEN = 0.4
    INTERACT_RADIUS = 30
    REPAIR_AMOUNT = 25
    ASTEROID_BASE_CHANCE = 0.008
    ASTEROID_CHANCE_INCREASE = 0.00005
    ASTEROID_DAMAGE = 35

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 20, bold=True)
        
        # Game Layout
        self._define_layout()

        # Initialize state variables
        self.player_pos = np.zeros(2, dtype=float)
        self.player_vel = np.zeros(2, dtype=float)
        self.systems = []
        self.resources = []
        self.upgrades = []
        self.particles = []
        self.player_resources = 0
        self.jetpack_fuel = 0.0
        self.player_upgrades = {}
        self.steps = 0
        self.score = 0.0
        self.last_space_held = False
        self.asteroid_impact_chance = 0.0

    def _define_layout(self):
        """Defines the static layout of the space station."""
        self.walls = [
            pygame.Rect(150, 100, 20, 200),
            pygame.Rect(470, 100, 20, 200),
            pygame.Rect(150, 100, 340, 20),
            pygame.Rect(150, 280, 340, 20),
        ]
        self.system_defs = [
            {'name': 'Life Support', 'pos': (100, 100), 'is_critical': True},
            {'name': 'Power Core',   'pos': (320, 200), 'is_critical': True},
            {'name': 'Navigation',   'pos': (540, 300), 'is_critical': True},
            {'name': 'Shield Gen',   'pos': (100, 300), 'is_critical': False},
            {'name': 'Comms Array',  'pos': (540, 100), 'is_critical': False},
        ]
        self.resource_defs = [
            {'pos': (220, 150)}, {'pos': (420, 250)}, {'pos': (50, 200)}
        ]
        self.upgrade_defs = [
            {'pos': (320, 50), 'type': 'fast_repair'},
            {'pos': (320, 350), 'type': 'jetpack_eff'}
        ]
        self.stars = [(random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)) for _ in range(150)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=float)
        self.player_vel = np.zeros(2, dtype=float)

        self.systems = []
        for s_def in self.system_defs:
            self.systems.append({
                'name': s_def['name'],
                'pos': np.array(s_def['pos'], dtype=float),
                'health': 100.0,
                'is_critical': s_def['is_critical'],
                'radius': 15
            })

        self.resources = [{'pos': np.array(r_def['pos'], dtype=float), 'active': True, 'radius': 8} for r_def in self.resource_defs]
        self.upgrades = [{'pos': np.array(u_def['pos'], dtype=float), 'type': u_def['type'], 'active': True, 'radius': 10} for u_def in self.upgrade_defs]

        self.player_resources = 3
        self.jetpack_fuel = self.JETPACK_FUEL_MAX
        self.player_upgrades = {'fast_repair': False, 'jetpack_eff': False}

        self.steps = 0
        self.score = 0.0
        self.asteroid_impact_chance = self.ASTEROID_BASE_CHANCE
        self.last_space_held = False
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.0
        
        # --- Update Game Logic ---
        reward += self._handle_player_movement(movement, shift_held)
        reward += self._handle_interaction(space_held)
        self._update_world_state()

        # --- Calculate Rewards & Termination ---
        for system in self.systems:
            if system['is_critical'] and system['health'] > 0:
                reward += 0.1

        self.score += reward
        self.steps += 1
        
        terminated = self._check_termination()
        
        if terminated:
            is_win = self.steps >= self.MAX_STEPS
            reward += 100.0 if is_win else -100.0
            self.score += 100.0 if is_win else -100.0

        self.last_space_held = space_held

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement, shift_held):
        accel = np.zeros(2, dtype=float)
        
        # Determine acceleration based on input
        if movement == 1: accel[1] -= 1  # Up
        elif movement == 2: accel[1] += 1  # Down
        elif movement == 3: accel[0] -= 1  # Left
        elif movement == 4: accel[0] += 1  # Right

        # Apply jetpack boost
        is_boosting = False
        if shift_held and self.jetpack_fuel > 0 and np.linalg.norm(accel) > 0:
            is_boosting = True
            accel_magnitude = self.PLAYER_JETPACK_ACCEL
            fuel_consumption = self.JETPACK_CONSUMPTION
            if self.player_upgrades['jetpack_eff']:
                fuel_consumption *= 0.6
            self.jetpack_fuel = max(0, self.jetpack_fuel - fuel_consumption)
            
            # Jetpack particles
            if self.steps % 2 == 0:
                p_vel = -accel * random.uniform(2, 4) + self.player_vel * 0.5
                self._add_particles(self.player_pos.copy(), 1, p_vel, 20, 4, self.COLOR_JETPACK)
        else:
            accel_magnitude = self.PLAYER_ACCEL

        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel) * accel_magnitude
        
        # Update velocity and position
        self.player_vel += accel
        self.player_vel *= self.PLAYER_DRAG
        
        # Clamp velocity
        vel_norm = np.linalg.norm(self.player_vel)
        if vel_norm > self.MAX_VEL:
            self.player_vel = self.player_vel / vel_norm * self.MAX_VEL

        self.player_pos += self.player_vel

        # Handle wall collisions
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
        for wall in self.walls:
            if player_rect.colliderect(wall):
                # Horizontal collision
                if player_rect.centerx < wall.centerx and self.player_vel[0] > 0: # Moving right into left side of wall
                    player_rect.right = wall.left
                    self.player_vel[0] *= -0.5
                elif player_rect.centerx > wall.centerx and self.player_vel[0] < 0: # Moving left into right side of wall
                    player_rect.left = wall.right
                    self.player_vel[0] *= -0.5
                # Vertical collision
                if player_rect.centery < wall.centery and self.player_vel[1] > 0: # Moving down into top side of wall
                    player_rect.bottom = wall.top
                    self.player_vel[1] *= -0.5
                elif player_rect.centery > wall.centery and self.player_vel[1] < 0: # Moving up into bottom side of wall
                    player_rect.top = wall.bottom
                    self.player_vel[1] *= -0.5
                self.player_pos = np.array(player_rect.center, dtype=float)


        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)
        
        # Regenerate jetpack fuel if not boosting
        if not is_boosting:
            self.jetpack_fuel = min(self.JETPACK_FUEL_MAX, self.jetpack_fuel + self.JETPACK_REGEN)
            
        return 0.0

    def _handle_interaction(self, space_held):
        reward = 0.0
        is_interacting = space_held and not self.last_space_held
        
        if not is_interacting:
            return reward

        # Find nearest interactable
        interactables = self.systems + self.resources + self.upgrades
        closest_obj = None
        min_dist = float('inf')

        for obj in interactables:
            # Skip inactive objects
            if 'active' in obj and not obj['active']:
                continue
            dist = np.linalg.norm(self.player_pos - obj['pos'])
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj
        
        if closest_obj and min_dist <= self.INTERACT_RADIUS:
            # Interact with a System
            if 'health' in closest_obj:
                if closest_obj['health'] < 100 and self.player_resources > 0:
                    self.player_resources -= 1
                    repair_val = self.REPAIR_AMOUNT * (1.5 if self.player_upgrades['fast_repair'] else 1.0)
                    closest_obj['health'] = min(100, closest_obj['health'] + repair_val)
                    reward += 1.0
                    # Repair sfx placeholder
                    self._add_particles(closest_obj['pos'], 10, None, 15, 3, self.COLOR_SYS_WARN)
            
            # Interact with a Resource
            elif 'active' in closest_obj and 'type' not in closest_obj:
                closest_obj['active'] = False
                self.player_resources += 1
                # Collect sfx placeholder
                self._add_particles(closest_obj['pos'], 8, None, 20, 4, self.COLOR_RESOURCE)
            
            # Interact with an Upgrade
            elif 'active' in closest_obj and 'type' in closest_obj:
                closest_obj['active'] = False
                self.player_upgrades[closest_obj['type']] = True
                reward += 5.0
                # Upgrade sfx placeholder
                self._add_particles(closest_obj['pos'], 15, None, 25, 5, self.COLOR_UPGRADE)
        
        return reward

    def _update_world_state(self):
        # Asteroid impacts
        if self.steps > 200:
            self.asteroid_impact_chance += self.ASTEROID_CHANCE_INCREASE
        
        for system in self.systems:
            if self.np_random.random() < self.asteroid_impact_chance:
                system['health'] = max(0, system['health'] - self.ASTEROID_DAMAGE)
                # Impact sfx placeholder
                self._add_particles(system['pos'], 20, None, 30, 5, self.COLOR_SYS_CRIT)

        # Respawn resources and upgrades
        if self.steps % 300 == 0 and self.steps > 0:
            for r in self.resources:
                if not r['active']:
                    r['active'] = True
                    break # Respawn one at a time
        if self.steps % 500 == 0 and self.steps > 0:
            for u in self.upgrades:
                if not u['active']:
                    u['active'] = True
                    break

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        
        critical_systems_failed = 0
        for system in self.systems:
            if system['is_critical'] and system['health'] <= 0:
                critical_systems_failed += 1
        
        num_critical_systems = sum(1 for s in self.systems if s['is_critical'])
        if critical_systems_failed == num_critical_systems:
            return True
            
        return False

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.player_resources}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star_pos in self.stars:
            size = random.choice([1, 1, 1, 2])
            pygame.draw.circle(self.screen, self.COLOR_STARS, star_pos, size/2)
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
    
    def _render_game_elements(self):
        # Systems
        for system in self.systems:
            x, y = int(system['pos'][0]), int(system['pos'][1])
            health_frac = system['health'] / 100.0
            
            if health_frac > 0.6: color = self.COLOR_SYS_HEALTHY
            elif health_frac > 0.2: color = self.COLOR_SYS_WARN
            else: color = self.COLOR_SYS_CRIT
            
            pygame.gfxdraw.filled_circle(self.screen, x, y, system['radius'], color)
            pygame.gfxdraw.aacircle(self.screen, x, y, system['radius'], color)
            
            # Health bar
            bar_w, bar_h = 40, 6
            bar_x, bar_y = x - bar_w / 2, y - system['radius'] - 12
            pygame.draw.rect(self.screen, self.COLOR_WALL, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_w * health_frac, bar_h))

        # Resources
        for resource in self.resources:
            if resource['active']:
                x, y = int(resource['pos'][0]), int(resource['pos'][1])
                r = resource['radius']
                points = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_RESOURCE)
                pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_RESOURCE)
        
        # Upgrades
        for upgrade in self.upgrades:
            if upgrade['active']:
                x, y = int(upgrade['pos'][0]), int(upgrade['pos'][1])
                r = upgrade['radius']
                pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_UPGRADE)
                pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_UPGRADE)
                # Draw a plus sign
                pygame.draw.line(self.screen, self.COLOR_BG, (x - r/2, y), (x + r/2, y), 2)
                pygame.draw.line(self.screen, self.COLOR_BG, (x, y - r/2), (x, y + r/2), 2)

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        r = self.PLAYER_SIZE
        
        # Glow
        pygame.gfxdraw.filled_circle(self.screen, x, y, int(r * 1.8), self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, x, y, int(r * 1.8), self.COLOR_PLAYER_GLOW)

        # Main Body
        pygame.gfxdraw.filled_circle(self.screen, x, y, r, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, x, y, r, self.COLOR_PLAYER)

    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'])
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 40), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # Score
        score_text = self.font_large.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Steps
        steps_text = self.font_large.render(f"CYCLE: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 10))

        # Resources
        res_text = self.font_large.render(f"REPAIR PARTS: {self.player_resources}", True, self.COLOR_TEXT)
        self.screen.blit(res_text, (200, 10))

        # Jetpack Fuel
        fuel_bar_w, fuel_bar_h = 100, 12
        fuel_bar_x, fuel_bar_y = 420, 14
        fuel_frac = self.jetpack_fuel / self.JETPACK_FUEL_MAX
        pygame.draw.rect(self.screen, self.COLOR_WALL, (fuel_bar_x, fuel_bar_y, fuel_bar_w, fuel_bar_h))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (fuel_bar_x, fuel_bar_y, fuel_bar_w * fuel_frac, fuel_bar_h))
        fuel_text = self.font_small.render("JETPACK", True, self.COLOR_TEXT)
        self.screen.blit(fuel_text, (fuel_bar_x - fuel_text.get_width() - 5, 12))

    def _add_particles(self, pos, count, base_vel, lifespan, size, color):
        for _ in range(count):
            if base_vel is not None:
                vel = base_vel + self.np_random.standard_normal(2) * 0.5
            else: # Explosion
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 4)
                vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            
            self.particles.append({
                'pos': pos.copy().astype(float),
                'vel': vel,
                'lifespan': self.np_random.integers(lifespan // 2, lifespan),
                'size': self.np_random.uniform(size * 0.5, size),
                'color': color
            })

    def close(self):
        pygame.quit()